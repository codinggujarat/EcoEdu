
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class CertificateService:
    def __init__(self, static_folder):
        self.static_folder = static_folder
        # We no longer use a template file, we create one from scratch.
        self.output_folder = os.path.join(static_folder, 'certificates')
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_certificate(self, user, milestone):
        try:
            print(f"DEBUG: Generating Custom Certificate for {user.name} at {milestone}")
            
            # 1. Create Blank Canvas (Landscape A4-ish High Res)
            width, height = 2000, 1400
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            
            # Colors
            color_green = "#22c55e"
            color_dark_green = "#14532d"
            color_gold = "#eab308"
            color_text = "#1f2937"
            
            # 2. Draw Borders
            # Main border
            margin = 60
            draw.rectangle(
                [(margin, margin), (width - margin, height - margin)], 
                outline=color_green, 
                width=15
            )
            
            # Inner Gold border
            margin_inner = 85
            draw.rectangle(
                [(margin_inner, margin_inner), (width - margin_inner, height - margin_inner)], 
                outline=color_gold, 
                width=5
            )

            # 3. Load Fonts
            try:
                # Big Title
                font_title = ImageFont.truetype("arialbd.ttf", 100) # Bold
                # Header
                font_header = ImageFont.truetype("arial.ttf", 60)
                # Name
                font_name = ImageFont.truetype("arialbd.ttf", 120)
                # Body
                font_body = ImageFont.truetype("arial.ttf", 50)
                # Small
                font_small = ImageFont.truetype("arial.ttf", 30)
                # Logo
                font_logo = ImageFont.truetype("arialbd.ttf", 70)
            except IOError:
                font_title = ImageFont.load_default()
                font_header = ImageFont.load_default()
                font_name = ImageFont.load_default()
                font_body = ImageFont.load_default()
                font_small = ImageFont.load_default()
                font_logo = ImageFont.load_default()

            # 4. Header Section
            # LOGO (Text representation)
            logo_text = "EcoEdu"
            bbox = draw.textbbox((0, 0), logo_text, font=font_logo)
            logo_w = bbox[2] - bbox[0]
            draw.text(((width - logo_w)/2, 150), logo_text, fill=color_green, font=font_logo)
            
            # "CERTIFICATE OF ACHIEVEMENT"
            title_text = "CERTIFICATE"
            subtitle_text = "OF ACHIEVEMENT"
            
            bbox = draw.textbbox((0, 0), title_text, font=font_title)
            title_w = bbox[2] - bbox[0]
            draw.text(((width - title_w)/2, 300), title_text, fill=color_dark_green, font=font_title)
            
            bbox = draw.textbbox((0, 0), subtitle_text, font=font_header)
            sub_w = bbox[2] - bbox[0]
            draw.text(((width - sub_w)/2, 420), subtitle_text, fill=color_gold, font=font_header)

            # 5. Body Section
            intro_text = "This certificate is proudly presented to"
            bbox = draw.textbbox((0, 0), intro_text, font=font_body)
            intro_w = bbox[2] - bbox[0]
            draw.text(((width - intro_w)/2, 600), intro_text, fill=color_text, font=font_body)
            
            # USER NAME
            name_text = user.name.upper()
            bbox = draw.textbbox((0, 0), name_text, font=font_name)
            name_w = bbox[2] - bbox[0]
            draw.text(((width - name_w)/2, 700), name_text, fill=color_dark_green, font=font_name)
            
            # Draw line under name
            line_w = max(name_w + 100, 800)
            draw.line(
                [(width/2 - line_w/2, 830), (width/2 + line_w/2, 830)], 
                fill=color_gold, 
                width=5
            )

            # MILESTONE TEXT
            milestone_text = f"For successfully verifying activities and achieving"
            milestone_points = f"{milestone} Eco-Points"
            
            bbox = draw.textbbox((0, 0), milestone_text, font=font_body)
            m_w = bbox[2] - bbox[0]
            draw.text(((width - m_w)/2, 900), milestone_text, fill=color_text, font=font_body)
            
            bbox = draw.textbbox((0, 0), milestone_points, font=font_title) # Recycle title font for emphasis
            mp_w = bbox[2] - bbox[0]
            draw.text(((width - mp_w)/2, 980), milestone_points, fill=color_green, font=font_title)

            # 6. Footer Section
            # Date
            date_str = datetime.now().strftime("%B %d, %Y")
            date_label = "DATE"
            
            # Signature
            sig_label = "SIGNATURE"
            sig_name = "EcoEdu Team"
            
            # Position: Date Left, Sig Right
            left_x = 400
            right_x = width - 400
            footer_y = 1200
            
            # Date Left
            draw.line([(left_x - 150, footer_y), (left_x + 150, footer_y)], fill=color_text, width=3)
            bbox = draw.textbbox((0, 0), date_str, font=font_small)
            d_w = bbox[2] - bbox[0]
            draw.text((left_x - d_w/2, footer_y - 40), date_str, fill=color_text, font=font_small)
            
            bbox = draw.textbbox((0, 0), date_label, font=font_small)
            dl_w = bbox[2] - bbox[0]
            draw.text((left_x - dl_w/2, footer_y + 10), date_label, fill="#6b7280", font=font_small)

            # Sig Right
            draw.line([(right_x - 150, footer_y), (right_x + 150, footer_y)], fill=color_text, width=3)
            # Fake signature font? Just italic/script if available, otherwise plain
            draw.text((right_x - 100, footer_y - 60), "EcoEdu Team", fill=color_dark_green, font=font_body) # Pseudo sig
            
            bbox = draw.textbbox((0, 0), sig_label, font=font_small)
            sl_w = bbox[2] - bbox[0]
            draw.text((right_x - sl_w/2, footer_y + 10), sig_label, fill="#6b7280", font=font_small)


            # Save
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"Certificate_{user.id}_{milestone}_{timestamp}.pdf"
            output_path = os.path.join(self.output_folder, filename)
            
            img.save(output_path, "PDF", resolution=100.0)
            
            print(f"DEBUG: Certified Saved: {output_path}")
            return filename
            
        except Exception as e:
            print(f"Certificate Gen Error: {e}")
            import traceback
            traceback.print_exc()
            return None
