"""
Centralized theme and color configuration.
Uses custom color palette with blues, beiges, and soft tones.
Supports both light and dark modes.
"""

# Light Mode Color Palette - Custom Colors
COLORS_LIGHT = {
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

# Dark Mode Color Palette
COLORS_DARK = {
    # Background colors
    'bg_primary': '#1a1a1a',      # Main background (dark)
    'bg_secondary': '#2d2d2d',    # Sidebar, subtle backgrounds
    'bg_tertiary': '#404040',     # Borders, dividers
    'bg_quaternary': '#2a2a2a',   # Input area
    
    # Text colors
    'text_primary': '#e0e0e0',    # Main text (light)
    'text_secondary': '#b0b0b0',  # Secondary text
    'text_tertiary': '#808080',   # Placeholders, muted text
    
    # Accent colors
    'accent_primary': '#67a2cb',  # Primary accent (medium blue - same as light)
    'accent_hover': '#5898b7',    # Accent hover state
    'accent_secondary': '#91b461', # Secondary accent (green - same as light)
    
    # Message colors
    'msg_user_bg': '#3a3a3a',     # User message background
    'msg_user_text': '#67a2cb',   # User message text (blue)
    'msg_assistant_bg': '#1a1a1a', # Assistant message background
    'msg_assistant_text': '#e0e0e0', # Assistant message text (light)
    
    # Status colors
    'status_connected': '#91b461', # Connected status (green - same as light)
    'status_disconnected': '#e0556a', # Disconnected status (red - same as light)
    
    # Button colors
    'btn_primary': '#67a2cb',     # Primary button (medium blue)
    'btn_primary_hover': '#5898b7', # Primary button hover
    'btn_danger': '#fd7589',      # Danger button (pink/red)
    'btn_danger_hover': '#e0556a', # Danger button hover (red)
    'btn_secondary': '#404040',   # Secondary button
    'btn_secondary_hover': '#505050', # Secondary button hover
    
    # Modal/overlay
    'modal_overlay': 'rgba(0,0,0,0.7)', # Modal backdrop
    'modal_bg': '#2d2d2d',        # Modal background
    
    # Scrollbar
    'scrollbar_track': '#2d2d2d', # Scrollbar track
    'scrollbar_thumb': '#505050', # Scrollbar thumb
    'scrollbar_thumb_hover': '#606060', # Scrollbar thumb hover
}

# Default to light mode
COLORS = COLORS_LIGHT

# UI Dimensions
DIMENSIONS = {
    'sidebar_width': '200px',
    'sidebar_collapsed_width': '50px',
    'border_radius': '8px',
    'border_radius_small': '6px',
    'message_max_width': '700px',  # More compact on desktop
    'input_padding': '10px 14px',
    'header_padding': '12px 20px',
}

# Font Settings
FONTS = {
    'family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    'size_base': '14px',
    'size_small': '12px',
    'size_large': '18px',
    'size_message': '14px',  # Same as base for compact UI
}

# Animation Settings
ANIMATIONS = {
    'transition_speed': '0.2s',
    'slide_duration': '0.3s',
}
