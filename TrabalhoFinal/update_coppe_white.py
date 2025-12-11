#!/usr/bin/env python3
import re
import os

def update_slide_to_white_coppe(filepath):
    """Update a slide to use white background with COPPE blue colors"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace dark background colors with white
    content = re.sub(r'background-color:\s*#0F172A', 'background-color: #FFFFFF', content)
    content = re.sub(r'background-color:\s*#000', 'background-color: #FFFFFF', content)
    
    # Replace background image with white background
    content = re.sub(
        r"background-image:\s*url\('[^']+'\);\s*background-size:\s*cover;\s*background-position:\s*center;",
        "background-color: #FFFFFF;",
        content
    )
    
    # Update text colors from white/light to dark blue
    content = re.sub(r'color:\s*#F8FAFC', 'color: #1a1a1a', content)
    content = re.sub(r'color:\s*#FFFFFF', 'color: #003d7a', content)
    content = re.sub(r'color:\s*#fff\b', 'color: #003d7a', content)
    
    # Update secondary text colors
    content = re.sub(r'color:\s*#CBD5E1', 'color: #333333', content)
    content = re.sub(r'color:\s*#94A3B8', 'color: #666666', content)
    content = re.sub(r'color:\s*#64748B', 'color: #888888', content)
    
    # Update cyan accent to COPPE blue
    content = re.sub(r'#38BDF8', '#005eb8', content)
    content = re.sub(r'#0EA5E9', '#005eb8', content)
    
    # Update purple/indigo to COPPE dark blue
    content = re.sub(r'#818CF8', '#003d7a', content)
    content = re.sub(r'#A78BFA', '#003d7a', content)
    
    # Remove dark overlay if present
    if '::before' in content and 'rgba(0, 0, 0' in content:
        # Remove the overlay CSS block
        content = re.sub(
            r'/\*\s*Dark overlay[^}]+\}\s*\}',
            '',
            content,
            flags=re.DOTALL
        )
    
    # Add COPPE top bar if not present
    if 'top-bar' not in content and '<body>' in content:
        top_bar_html = '''
        <!-- COPPE Top Bar -->
        <div class="top-bar"></div>
'''
        top_bar_css = '''
        /* COPPE Top Bar */
        .top-bar {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, #003d7a 0%, #005eb8 100%);
            z-index: 100;
        }
'''
        # Add CSS
        content = content.replace('</style>', top_bar_css + '    </style>')
        # Add HTML after slide-container opening
        content = content.replace('<div class="slide-container">', '<div class="slide-container">' + top_bar_html, 1)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

# Update slides 2-21
for i in range(2, 22):
    filepath = f'/home/ubuntu/pneumonia_project/presentation/slide_{i}.html'
    if os.path.exists(filepath):
        try:
            update_slide_to_white_coppe(filepath)
            print(f'✓ Updated slide_{i}.html')
        except Exception as e:
            print(f'✗ Error updating slide_{i}.html: {e}')
    else:
        print(f'- Skipped slide_{i}.html (not found)')

print('\n✓ All slides updated to COPPE white background theme!')
