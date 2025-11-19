"""
Centralized theme and color configuration.
Monochrome light theme with dark text on light background.
"""

# Color Palette - Monochrome Light Theme
COLORS = {
    # Background colors
    'bg_primary': '#ffffff',      # Main background (white)
    'bg_secondary': '#f5f5f5',    # Sidebar, subtle backgrounds
    'bg_tertiary': '#e0e0e0',     # Borders, dividers
    'bg_quaternary': '#fafafa',   # Input area
    
    # Text colors
    'text_primary': '#1a1a1a',    # Main text (dark)
    'text_secondary': '#666666',   # Secondary text
    'text_tertiary': '#999999',   # Placeholders, muted text
    
    # Accent colors (subtle grays)
    'accent_primary': '#333333',  # Primary accent (dark gray)
    'accent_hover': '#000000',    # Accent hover state (black)
    'accent_secondary': '#4a4a4a', # Secondary accent
    
    # Message colors
    'msg_user_bg': '#f0f0f0',     # User message background (light gray)
    'msg_user_text': '#1a1a1a',   # User message text (dark)
    'msg_assistant_bg': '#ffffff', # Assistant message background (white)
    'msg_assistant_text': '#1a1a1a', # Assistant message text (dark)
    
    # Status colors
    'status_connected': '#4a4a4a', # Connected status
    'status_disconnected': '#999999', # Disconnected status
    
    # Button colors
    'btn_primary': '#1a1a1a',     # Primary button (dark)
    'btn_primary_hover': '#000000', # Primary button hover (black)
    'btn_danger': '#666666',      # Danger button
    'btn_danger_hover': '#4a4a4a', # Danger button hover
    'btn_secondary': '#e0e0e0',   # Secondary button
    'btn_secondary_hover': '#d0d0d0', # Secondary button hover
    
    # Modal/overlay
    'modal_overlay': 'rgba(0,0,0,0.5)', # Modal backdrop
    'modal_bg': '#ffffff',        # Modal background
    
    # Scrollbar
    'scrollbar_track': '#f5f5f5', # Scrollbar track
    'scrollbar_thumb': '#d0d0d0', # Scrollbar thumb
    'scrollbar_thumb_hover': '#b0b0b0', # Scrollbar thumb hover
}

# UI Dimensions
DIMENSIONS = {
    'sidebar_width': '240px',
    'sidebar_collapsed_width': '60px',
    'border_radius': '12px',
    'border_radius_small': '8px',
    'message_max_width': '900px',  # Wider for full-width feel
    'input_padding': '12px 16px',
    'header_padding': '16px 24px',
}

# Font Settings
FONTS = {
    'family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    'size_base': '16px',
    'size_small': '14px',
    'size_large': '20px',
    'size_message': '18px',  # Larger text for messages
}

# Animation Settings
ANIMATIONS = {
    'transition_speed': '0.2s',
    'slide_duration': '0.3s',
}
