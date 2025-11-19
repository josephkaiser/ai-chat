"""
Centralized theme and color configuration.
Modify these values to customize the appearance across the entire application.
"""

# Color Palette
COLORS = {
    # Background colors
    'bg_primary': '#0f172a',      # Main background
    'bg_secondary': '#1e293b',     # Sidebar, header, input area
    'bg_tertiary': '#334155',      # Borders, hover states
    
    # Text colors
    'text_primary': '#f1f5f9',     # Main text
    'text_secondary': '#94a3b8',    # Secondary text, placeholders
    
    # Accent colors
    'accent_primary': '#10b981',   # Primary accent (green)
    'accent_hover': '#059669',      # Accent hover state
    'accent_secondary': '#2563eb',  # Secondary accent (blue)
    
    # Message colors
    'msg_user_bg': '#2563eb',      # User message background
    'msg_user_text': '#ffffff',    # User message text
    'msg_assistant_bg': '#1e293b', # Assistant message background
    'msg_assistant_text': '#f1f5f9', # Assistant message text
    
    # Status colors
    'status_connected': '#10b981', # Connected status
    'status_disconnected': '#ef4444', # Disconnected status
    
    # Button colors
    'btn_primary': '#10b981',      # Primary button
    'btn_primary_hover': '#059669', # Primary button hover
    'btn_danger': '#ef4444',       # Danger button (delete)
    'btn_danger_hover': '#dc2626',  # Danger button hover
    'btn_secondary': '#334155',    # Secondary button
    'btn_secondary_hover': '#475569', # Secondary button hover
    
    # Modal/overlay
    'modal_overlay': 'rgba(0,0,0,0.8)', # Modal backdrop
    'modal_bg': '#1e293b',         # Modal background
    
    # Scrollbar
    'scrollbar_track': '#1e293b',  # Scrollbar track
    'scrollbar_thumb': '#334155',   # Scrollbar thumb
    'scrollbar_thumb_hover': '#475569', # Scrollbar thumb hover
}

# UI Dimensions
DIMENSIONS = {
    'sidebar_width': '280px',
    'border_radius': '8px',
    'border_radius_small': '6px',
    'message_max_width': '70%',
    'input_padding': '15px',
    'header_padding': '20px 30px',
}

# Font Settings
FONTS = {
    'family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    'size_base': '14px',
    'size_small': '13px',
    'size_large': '24px',
}

# Animation Settings
ANIMATIONS = {
    'transition_speed': '0.2s',
    'slide_duration': '0.3s',
}

