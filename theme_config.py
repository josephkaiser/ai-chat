"""
Centralized theme and color configuration.
Uses custom color palette with blues, beiges, and soft tones.
"""

# Color Palette - Custom Colors
COLORS = {
    # Background colors
    'bg_primary': '#fffefb',      # Main background (almost white from palette 3)
    'bg_secondary': '#ececdf',    # Sidebar, subtle backgrounds (from palette 2)
    'bg_tertiary': '#e0e0e0',     # Borders, dividers (neutral gray)
    'bg_quaternary': '#ffeae7',   # Input area (very light pink from palette 3)
    
    # Text colors
    'text_primary': '#366c9c',    # Main text (dark blue from palette 1)
    'text_secondary': '#5898b7',  # Secondary text (blue from palette 2)
    'text_tertiary': '#9dceda',   # Placeholders, muted text (light blue from palette 2)
    
    # Accent colors
    'accent_primary': '#67a2cb',  # Primary accent (medium blue from palette 1)
    'accent_hover': '#5898b7',    # Accent hover state (blue from palette 2)
    'accent_secondary': '#91b461', # Secondary accent (green from palette 2)
    
    # Message colors
    'msg_user_bg': '#f5e3d3',     # User message background (light beige from palette 1)
    'msg_user_text': '#366c9c',   # User message text (dark blue)
    'msg_assistant_bg': '#fffefb', # Assistant message background (almost white)
    'msg_assistant_text': '#366c9c', # Assistant message text (dark blue)
    
    # Status colors
    'status_connected': '#91b461', # Connected status (green from palette 2)
    'status_disconnected': '#e0556a', # Disconnected status (red from palette 3)
    
    # Button colors
    'btn_primary': '#67a2cb',     # Primary button (medium blue)
    'btn_primary_hover': '#5898b7', # Primary button hover
    'btn_danger': '#fd7589',      # Danger button (pink/red from palette 3)
    'btn_danger_hover': '#e0556a', # Danger button hover (red)
    'btn_secondary': '#a6d4f2',   # Secondary button (light blue from palette 1)
    'btn_secondary_hover': '#9dceda', # Secondary button hover (light blue from palette 2)
    
    # Modal/overlay
    'modal_overlay': 'rgba(54,108,156,0.5)', # Modal backdrop (dark blue with transparency)
    'modal_bg': '#fffefb',        # Modal background
    
    # Scrollbar
    'scrollbar_track': '#ececdf', # Scrollbar track (light beige from palette 2)
    'scrollbar_thumb': '#a6d4f2', # Scrollbar thumb (light blue)
    'scrollbar_thumb_hover': '#9dceda', # Scrollbar thumb hover
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
