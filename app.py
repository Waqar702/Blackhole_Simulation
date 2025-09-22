"""
Main application entry point for black hole simulation.

This module provides the main application class that orchestrates
all components of the simulation.
"""

import argparse
import logging
import sys
from typing import Optional

from .demos import Demo2D, Demo3D


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Black Hole Simulation - Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m blackhole_python.app 2d          # Run 2D lensing demo
  python -m blackhole_python.app 3d          # Run 3D GPU simulation
  python -m blackhole_python.app --help      # Show this help message
        """
    )
    
    parser.add_argument(
        'demo',
        choices=['2d', '3d'],
        help='Demo to run (2d for lensing, 3d for GPU simulation)',
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=800,
        help='Window width in pixels (default: 800)',
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=600,
        help='Window height in pixels (default: 600)',
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)',
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Black Hole Simulation v0.1.0',
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting {args.demo.upper()} demo")
        logger.info(f"Window size: {args.width}x{args.height}")
        
        # Create and run demo
        if args.demo == '2d':
            demo = Demo2D(width=args.width, height=args.height)
        else:  # 3d
            demo = Demo3D(width=args.width, height=args.height)
        
        demo.run()
        
        logger.info("Demo completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
