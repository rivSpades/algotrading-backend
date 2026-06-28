"""Live Trading Admin."""

from django.contrib import admin
from django.db.models import DateTimeField, Q
from django.db.models.functions import Coalesce
from django.urls import reverse
from django.utils.html import escape, format_html
from django.utils.safestring import mark_safe

from .models import (
    Broker,
    SymbolBrokerAssociation,
    StrategyDeployment,
    DeploymentSymbol,
    DeploymentEvent,
    LiveTrade,
)


@admin.register(Broker)
class BrokerAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'paper_trading_active', 'real_money_active', 'created_at']
    list_filter = ['paper_trading_active', 'real_money_active', 'created_at']
    search_fields = ['name', 'code']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(SymbolBrokerAssociation)
class SymbolBrokerAssociationAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'broker', 'long_active', 'short_active', 'verified_at', 'updated_at']
    list_filter = ['broker', 'long_active', 'short_active', 'verified_at']
    search_fields = ['symbol__ticker', 'broker__name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(StrategyDeployment)
class StrategyDeploymentAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', 'strategy', 'parameter_set', 'broker',
        'deployment_type', 'position_mode', 'status', 'created_at',
    ]
    list_filter = ['deployment_type', 'status', 'position_mode', 'broker', 'strategy']
    search_fields = ['name', 'strategy__name', 'broker__name', 'parameter_set__signature']
    raw_id_fields = ['strategy', 'parameter_set', 'broker', 'parent_deployment']
    readonly_fields = [
        'created_at', 'updated_at', 'started_at', 'activated_at',
        'evaluated_at', 'last_signal_at',
    ]
    ordering = ['-created_at']


@admin.register(DeploymentSymbol)
class DeploymentSymbolAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'deployment', 'symbol', 'position_mode', 'status',
        'tier', 'color_overall', 'priority', 'updated_at',
    ]
    list_filter = ['status', 'tier', 'color_overall', 'position_mode']
    search_fields = ['symbol__ticker', 'deployment__name']
    raw_id_fields = ['deployment', 'symbol']
    readonly_fields = ['created_at', 'updated_at', 'last_signal_at', 'last_evaluated_at']
    ordering = ['deployment', 'priority']


@admin.register(DeploymentEvent)
class DeploymentEventAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'deployment', 'deployment_symbol', 'event_type',
        'level', 'actor_type', 'actor_id', 'created_at',
    ]
    list_filter = ['event_type', 'level', 'actor_type']
    search_fields = ['deployment__name', 'message', 'actor_id']
    raw_id_fields = ['deployment', 'deployment_symbol']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def get_search_results(self, request, queryset, search_term):
        """Also match ``context.live_trade_id`` / ``main_live_trade_id`` when search is numeric."""
        qs, use_distinct = super().get_search_results(request, queryset, search_term)
        term = (search_term or '').strip()
        if not term.isdigit():
            return qs, use_distinct
        tid = int(term)
        base = self.get_queryset(request)
        json_hits = base.filter(
            Q(context__live_trade_id=tid)
            | Q(context__live_trade_id=str(tid))
            | Q(context__main_live_trade_id=tid)
            | Q(context__main_live_trade_id=str(tid))
            | Q(context__hedge_leg_live_trade_ids__contains=[tid])
        )
        return (qs | json_hits).distinct(), True


def _live_trade_meta_is_hedge(obj: LiveTrade) -> bool:
    return bool((obj.metadata or {}).get('is_hedge_leg'))


def _live_trade_pk_from_metadata(val):
    """Coerce stored JSON hedge parent/id to integer LiveTrade pk for admin URLs."""
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(str(val).strip())
        except ValueError:
            return None


class LiveTradeHedgeLegFilter(admin.SimpleListFilter):
    """Overlay legs (VIXY/VIXM) vs strategy sleeve mains."""

    title = 'vol hedge leg'
    parameter_name = 'vol_hedge_leg'

    def lookups(self, request, model_admin):
        return (
            ('yes', 'Vol hedge only (has parent trade)'),
            ('no', 'Main / standalone (no vol overlay marker)'),
        )

    def queryset(self, request, queryset):
        v = self.value()
        if v == 'yes':
            return queryset.filter(metadata__is_hedge_leg=True)
        if v == 'no':
            return queryset.filter(
                Q(metadata__is_hedge_leg__isnull=True)
                | Q(metadata__is_hedge_leg=False)
            )
        return queryset


class LiveTradeLedgerFilter(admin.SimpleListFilter):
    """Filter by whether a row has an exit timestamp (closed) vs still open.

    Note: this only filters **LiveTrade** ledger rows. Order/exit **audit lines** live under
    **Deployment events** (or the “Audit events” block on a trade’s change page).
    """

    title = 'ledger (exit)'
    parameter_name = 'ledger_exit'

    def lookups(self, request, model_admin):
        return (
            ('open', 'Open · no exit time'),
            ('closed', 'Closed · has exit time'),
        )

    def queryset(self, request, queryset):
        v = self.value()
        if v == 'open':
            return queryset.filter(exit_timestamp__isnull=True)
        if v == 'closed':
            return queryset.filter(exit_timestamp__isnull=False)
        return queryset


@admin.register(LiveTrade)
class LiveTradeAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'deployment',
        'symbol',
        '_paired_trade_linkage',
        '_hedge_icon',
        'trade_type',
        'position_mode',
        'entry_price',
        'exit_price',
        'quantity',
        'pnl',
        'is_winner',
        'status',
        'entry_timestamp',
        'exit_timestamp',
        'updated_at',
    ]
    list_filter = [
        LiveTradeLedgerFilter,
        LiveTradeHedgeLegFilter,
        'status',
        'trade_type',
        'position_mode',
        'deployment',
        'is_winner',
        'deployment__strategy',
        'deployment__deployment_type',
    ]
    search_fields = [
        'symbol__ticker',
        'deployment__name',
        'broker_order_id',
    ]
    raw_id_fields = ['deployment', 'deployment_symbol', 'symbol']
    readonly_fields = [
        '_paired_trade_linkage',
        '_recent_deployment_events',
        'created_at',
        'updated_at',
        'broker_order_id',
        'pnl_percentage',
        'is_winner',
        'entry_timestamp',
        'exit_timestamp',
    ]
    fieldsets = (
        ('Placement', {
            'fields': (
                'deployment',
                'deployment_symbol',
                'symbol',
                '_paired_trade_linkage',
                'position_mode',
                'trade_type',
            ),
        }),
        ('Prices', {
            'fields': ('entry_price', 'exit_price', 'quantity'),
        }),
        ('Timestamps', {
            'fields': ('entry_timestamp', 'exit_timestamp'),
        }),
        (
            'Results',
            {
                'description': (
                    'Per-trade max drawdown is not persisted on live rows (unlike backtest `Trade`); '
                    'the deployment UI shows an em dash in that column.'
                ),
                'fields': ('pnl', 'pnl_percentage', 'is_winner', 'status'),
            },
        ),
        ('Broker', {
            'fields': ('broker_order_id',),
        }),
        (
            'Exit / order audit (DeploymentEvent)',
            {
                'description': (
                    'The changelist is only LiveTrade ledger rows — not separate event lines per exit. '
                    'Use the table below or Admin → Deployment events '
                    '(search by numeric LiveTrade id to match context). '
                    'Sidebar filters “vol hedge leg” → “Main / standalone” hides vol legs; '
                    '“ledger (exit)” → “Open” hides closed trades.'
                ),
                'fields': ('_recent_deployment_events',),
            },
        ),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',),
        }),
        ('Audit', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    date_hierarchy = 'updated_at'
    list_select_related = ('deployment', 'symbol', 'deployment_symbol')
    show_full_result_count = False

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.annotate(
            activity_ts=Coalesce('exit_timestamp', 'entry_timestamp', output_field=DateTimeField()),
        ).order_by('-activity_ts', '-id')

    @admin.display(description='Linked trade')
    def _paired_trade_linkage(self, obj):
        md = obj.metadata or {}

        def trade_change_url(pk: int) -> str:
            return reverse('admin:live_trading_livetrade_change', args=[pk])

        if md.get('is_hedge_leg'):
            parent_id = _live_trade_pk_from_metadata(md.get('hedge_parent_live_trade_id'))
            parent_t = md.get('hedge_parent_ticker') or '?'
            if parent_id:
                url = trade_change_url(parent_id)
                return format_html(
                    'Parent sleeve: <a href="{}">LiveTrade&nbsp;#{}</a> · {}',
                    url,
                    parent_id,
                    parent_t,
                )
            return format_html(
                '<span title="missing hedge_parent_live_trade_id">Parent unknown · {}</span>',
                parent_t,
            )

        hid = md.get('hedge_leg_live_trade_id')
        htk = md.get('hedge_leg_ticker')
        hk = _live_trade_pk_from_metadata(hid)
        if hk:
            url = trade_change_url(hk)
            label = htk or 'vol hedge'
            return format_html(
                'Vol hedge: <a href="{}">LiveTrade&nbsp;#{}</a> · {}',
                url,
                hk,
                label,
            )
        return format_html('&mdash;')

    @admin.display(description='H', boolean=True)
    def _hedge_icon(self, obj):
        return _live_trade_meta_is_hedge(obj)

    @admin.display(description='Related deployment events')
    def _recent_deployment_events(self, obj: LiveTrade):
        if not obj.pk:
            return format_html('&mdash;')
        q_ev = (
            Q(context__live_trade_id=obj.pk)
            | Q(context__live_trade_id=str(obj.pk))
            | Q(context__main_live_trade_id=obj.pk)
            | Q(context__main_live_trade_id=str(obj.pk))
            | Q(context__hedge_leg_live_trade_ids__contains=[obj.pk])
        )
        rows = list(
            DeploymentEvent.objects.filter(deployment_id=obj.deployment_id)
            .filter(q_ev)
            .order_by('-created_at')[:40]
        )
        if not rows:
            de_list = reverse('admin:live_trading_deploymentevent_changelist')
            return format_html(
                '<p class="help">No events whose <code>context</code> references this trade id. '
                'Try <a href="{}?deployment__id__exact={}">deployment-scoped events</a> '
                'or search events by this id: <code>{}</code>.</p>',
                de_list,
                obj.deployment_id,
                obj.pk,
            )
        parts = [
            '<thead><tr><th>When</th><th>Type</th><th>Level</th><th>Message</th></tr></thead><tbody>'
        ]
        for e in rows:
            url = reverse('admin:live_trading_deploymentevent_change', args=[e.pk])
            parts.append(
                '<tr><td>{}</td><td><a href="{}">{}</a></td><td>{}</td><td>{}</td></tr>'.format(
                    escape(e.created_at.isoformat(sep=' ', timespec='seconds')),
                    url,
                    escape(e.event_type),
                    escape(e.level),
                    escape((e.message or '')[:240]),
                )
            )
        parts.append('</tbody>')
        return format_html(
            '<table class="listing" style="width:100%">{}</table>',
            mark_safe(''.join(parts)),
        )
