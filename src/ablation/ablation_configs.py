ABLATION_CONFIGS = {
    'full_system': {
        'skip_frames': True,
        'trend_analysis': True,
        'adaptive_threshold': True,
        'temporal_smoothing': True,
        'description': 'Complete proposed system'
    },
    
    'no_skip': {
        'skip_frames': False,  # Run YOLO every frame
        'trend_analysis': True,
        'adaptive_threshold': True,
        'temporal_smoothing': True,
        'description': 'Full inference (no Skip-3)'
    },
    
    'no_trend': {
        'skip_frames': True,
        'trend_analysis': False,  # Static overlap only
        'adaptive_threshold': True,
        'temporal_smoothing': True,
        'description': 'No temporal trend analysis'
    },
    
    'no_adaptive': {
        'skip_frames': True,
        'trend_analysis': True,
        'adaptive_threshold': False,  # Fixed 10% threshold
        'temporal_smoothing': True,
        'description': 'Fixed overlap threshold'
    },
    
    'no_smoothing': {
        'skip_frames': True,
        'trend_analysis': True,
        'adaptive_threshold': True,
        'temporal_smoothing': False,  # Raw Hough lines
        'description': 'No lane temporal smoothing'
    },
    
    'minimal': {
        'skip_frames': False,
        'trend_analysis': False,
        'adaptive_threshold': False,
        'temporal_smoothing': False,
        'description': 'Naive baseline (all features off)'
    }
}
