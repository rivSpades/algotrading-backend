"""
Evaluation Service for Live Trading Deployments
Handles paper trading evaluation criteria checking
"""

from decimal import Decimal
from django.utils import timezone
from typing import Dict, Optional
from ..models import LiveTradingDeployment, LiveTrade
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating live trading deployments against criteria"""
    
    @staticmethod
    def evaluate_deployment(deployment: LiveTradingDeployment) -> Dict:
        """
        Evaluate a deployment against its evaluation criteria
        
        Args:
            deployment: LiveTradingDeployment instance to evaluate
        
        Returns:
            Dict with evaluation results:
            {
                'trades_count': int,
                'total_pnl': float,
                'sharpe_ratio': float,
                'passed': bool,
                'evaluated_at': datetime,
                'criteria_met': {
                    'min_trades': bool,
                    'min_sharpe_ratio': bool,
                    'min_pnl': bool
                }
            }
        """
        criteria = deployment.evaluation_criteria
        min_trades = criteria.get('min_trades', 0)
        min_sharpe_ratio = criteria.get('min_sharpe_ratio', 1.0)
        min_pnl = criteria.get('min_pnl', 0.0)
        
        # Get closed trades for evaluation
        closed_trades = deployment.live_trades.filter(status='closed')
        trades_count = closed_trades.count()
        
        # Calculate total PnL
        total_pnl = sum(float(trade.pnl or 0) for trade in closed_trades)
        
        # Calculate Sharpe ratio from closed trades
        sharpe_ratio = EvaluationService._calculate_sharpe_ratio(closed_trades)
        
        # Check criteria
        criteria_met = {
            'min_trades': trades_count >= min_trades,
            'min_sharpe_ratio': sharpe_ratio > min_sharpe_ratio,  # Must be strictly greater
            'min_pnl': total_pnl > min_pnl  # Must be strictly greater than 0
        }
        
        # All criteria must be met
        passed = all(criteria_met.values())
        
        results = {
            'trades_count': trades_count,
            'total_pnl': float(total_pnl),
            'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio is not None else None,
            'passed': passed,
            'evaluated_at': timezone.now().isoformat(),
            'criteria_met': criteria_met,
            'criteria': {
                'min_trades': min_trades,
                'min_sharpe_ratio': min_sharpe_ratio,
                'min_pnl': min_pnl
            }
        }
        
        logger.info(
            f"Evaluation for deployment {deployment.id}: "
            f"trades={trades_count}/{min_trades}, "
            f"sharpe={sharpe_ratio:.4f}>{min_sharpe_ratio}, "
            f"pnl={total_pnl:.2f}>{min_pnl}, "
            f"passed={passed}"
        )
        
        return results
    
    @staticmethod
    def _calculate_sharpe_ratio(trades) -> Optional[float]:
        """
        Calculate Sharpe ratio from a list of trades
        
        Sharpe Ratio = (Mean Return) / (Standard Deviation of Returns) * sqrt(252)
        Assumes daily returns (annualized by multiplying by sqrt(252))
        
        Args:
            trades: QuerySet or list of LiveTrade instances
        
        Returns:
            Sharpe ratio as float, or None if insufficient data
        """
        if not trades:
            return None
        
        # Get PnL values
        pnl_values = [float(trade.pnl or 0) for trade in trades if trade.pnl is not None]
        
        if len(pnl_values) < 2:
            # Need at least 2 trades to calculate standard deviation
            return None
        
        # Calculate returns (PnL as percentage of entry value)
        # For simplicity, we'll use PnL directly, assuming normalized returns
        # In a real implementation, you'd want to normalize by trade size
        returns = np.array(pnl_values)
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Sample standard deviation
        
        if std_return == 0:
            # No variance, return None or 0
            return 0.0 if mean_return == 0 else None
        
        # Calculate Sharpe ratio (annualized, assuming daily data)
        # Annualized Sharpe = (Mean Daily Return / Std Daily Return) * sqrt(252)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return float(sharpe_ratio)
    
    @staticmethod
    def check_and_update_evaluation(deployment: LiveTradingDeployment) -> Optional[Dict]:
        """
        Check if deployment has reached minimum trades threshold and evaluate if so
        
        Args:
            deployment: LiveTradingDeployment instance
        
        Returns:
            Evaluation results dict if evaluation was performed, None otherwise
        """
        if deployment.deployment_type != 'paper':
            # Only evaluate paper trading deployments
            return None
        
        if deployment.status not in ['evaluating', 'active']:
            # Only evaluate if deployment is running
            return None
        
        criteria = deployment.evaluation_criteria
        min_trades = criteria.get('min_trades', 0)
        
        # Count closed trades
        closed_trades_count = deployment.live_trades.filter(status='closed').count()
        
        if closed_trades_count < min_trades:
            # Not enough trades yet
            logger.debug(
                f"Deployment {deployment.id} has {closed_trades_count}/{min_trades} trades, "
                "not ready for evaluation"
            )
            return None
        
        # Check if already evaluated
        if deployment.evaluated_at:
            logger.debug(f"Deployment {deployment.id} already evaluated")
            return deployment.evaluation_results
        
        # Perform evaluation
        results = EvaluationService.evaluate_deployment(deployment)
        
        # Update deployment
        deployment.evaluation_results = results
        deployment.evaluated_at = timezone.now()
        
        if results['passed']:
            deployment.status = 'passed'
            logger.info(f"Deployment {deployment.id} passed evaluation")
        else:
            deployment.status = 'failed'
            logger.info(f"Deployment {deployment.id} failed evaluation")
        
        deployment.save()
        
        return results



