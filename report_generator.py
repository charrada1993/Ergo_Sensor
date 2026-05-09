"""
report_generator.py
====================
Professional medical-grade PDF report for Ergo Sensor.
Uses ReportLab Platypus (flowable layout).
"""

import os
import io
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable

# Optional AI engine (graceful if not available)
try:
    from ai_engine import AIModels
    _AI_AVAILABLE = True
except Exception:
    _AI_AVAILABLE = False


# =============================================================================
# Colour palette
# =============================================================================
C_NAVY      = colors.HexColor('#0F172A') # Dark slate navy
C_TEAL      = colors.HexColor('#00E5FF') # Cyber cyan
C_LIGHT     = colors.HexColor('#E8F4F8')
C_ACCENT    = colors.HexColor('#FF4D6D') # Danger red
C_WARN      = colors.HexColor('#FFAA00') # Warn orange
C_OK        = colors.HexColor('#00E5A0') # OK Green
C_BORDER    = colors.HexColor('#E2E8F0') # Light gray border
C_WHITE     = colors.white
C_BLACK     = colors.black
C_GREY_DARK = colors.HexColor('#334155') # Text gray
C_GREY_MID  = colors.HexColor('#64748B')
C_GREY_LITE = colors.HexColor('#F8FAFC') # Background light
C_TABLE_ALT = colors.HexColor('#F1F5F9') # Zebra striping


# =============================================================================
# Custom flowable: colour header band
# =============================================================================
class SectionHeader(Flowable):
    def __init__(self, text, width=None, bg=C_NAVY, fg=C_WHITE, font_size=12):
        super().__init__()
        self._text      = text
        self._width     = width or (A4[0] - 4 * cm)
        self._bg        = bg
        self._fg        = fg
        self._font_size = font_size
        self.height     = font_size + 10

    def draw(self):
        self.canv.setFillColor(self._bg)
        self.canv.rect(0, 0, self._width, self.height, fill=1, stroke=0)
        self.canv.setFillColor(self._fg)
        self.canv.setFont('Helvetica-Bold', self._font_size)
        self.canv.drawString(6, 3, self._text)


# =============================================================================
# Page template callbacks (header / footer)
# =============================================================================
def _on_first_page(canvas, doc):
    _draw_header_footer(canvas, doc, is_first=True)

def _on_later_pages(canvas, doc):
    _draw_header_footer(canvas, doc, is_first=False)

def _draw_header_footer(canvas, doc, is_first):
    w, h = A4
    canvas.saveState()
    # top band
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, h - 1.5 * cm, w, 1.5 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont('Helvetica-Bold', 12)
    canvas.drawString(1.5 * cm, h - 1.05 * cm, 'ERGO SENSOR')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(C_TEAL)
    canvas.drawRightString(w - 1.5 * cm, h - 1.05 * cm,
                           'Musculoskeletal Disorder Risk Assessment')
    # accent stripe
    canvas.setFillColor(C_TEAL)
    canvas.rect(0, h - 1.6 * cm, w, 0.1 * cm, fill=1, stroke=0)
    
    # footer
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, w, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_GREY_MID)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(1.5 * cm, 0.45 * cm,
                      f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  CONFIDENTIAL - For Medical Use Only')
    canvas.setFillColor(C_WHITE)
    canvas.drawRightString(w - 1.5 * cm, 0.45 * cm,
                           f'Page {doc.page}')
    canvas.restoreState()


# =============================================================================
# Styles
# =============================================================================
def _build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles['title'] = ParagraphStyle(
        'ReportTitle', parent=base['Normal'],
        fontSize=22, textColor=C_NAVY, fontName='Helvetica-Bold',
        spaceAfter=4, alignment=TA_CENTER
    )
    styles['subtitle'] = ParagraphStyle(
        'ReportSubtitle', parent=base['Normal'],
        fontSize=11, textColor=C_TEAL, fontName='Helvetica',
        spaceAfter=2, alignment=TA_CENTER
    )
    styles['body'] = ParagraphStyle(
        'Body', parent=base['Normal'],
        fontSize=9, textColor=C_GREY_DARK, spaceAfter=6, leading=14
    )
    styles['bold'] = ParagraphStyle(
        'Bold', parent=base['Normal'],
        fontSize=9, fontName='Helvetica-Bold', textColor=C_GREY_DARK
    )
    styles['small'] = ParagraphStyle(
        'Small', parent=base['Normal'],
        fontSize=8, textColor=C_GREY_MID, leading=12
    )
    styles['alert_red'] = ParagraphStyle(
        'AlertRed', parent=base['Normal'],
        fontSize=9, textColor=C_ACCENT, fontName='Helvetica-Bold'
    )
    styles['alert_ok'] = ParagraphStyle(
        'AlertOk', parent=base['Normal'],
        fontSize=9, textColor=C_OK, fontName='Helvetica-Bold'
    )
    styles['cell_hdr'] = ParagraphStyle(
        'CellHdr', parent=base['Normal'],
        fontSize=8, fontName='Helvetica-Bold',
        textColor=C_WHITE, alignment=TA_CENTER
    )
    styles['cell'] = ParagraphStyle(
        'Cell', parent=base['Normal'],
        fontSize=8, alignment=TA_CENTER
    )
    return styles


# =============================================================================
# Helper utilities
# =============================================================================
def _rula_colour(val):
    if val >= 7: return C_ACCENT
    if val >= 5: return C_WARN
    return C_OK

def _reba_colour(val):
    if val >= 11: return C_ACCENT
    if val >= 8: return C_WARN
    if val >= 4: return C_WARN
    return C_OK

def _fig_to_image(fig, width=15 * cm, height=7 * cm):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height)

def _styled_table(data_rows, col_widths, hdr_bg=C_GREY_LITE):
    t = Table(data_rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), hdr_bg),
        ('TEXTCOLOR',    (0, 0), (-1, 0), C_NAVY),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0), 9),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING',(0, 0), (-1, 0), 8),
        ('TOPPADDING',   (0, 0), (-1, 0), 8),
        ('LINEBELOW',    (0, 0), (-1, 0), 1.5, C_TEAL),
        
        ('FONTNAME',     (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',     (0, 1), (-1, -1), 9),
        ('TEXTCOLOR',    (0, 1), (-1, -1), C_GREY_DARK),
        ('BOTTOMPADDING',(0, 1), (-1, -1), 6),
        ('TOPPADDING',   (0, 1), (-1, -1), 6),
        ('LINEBELOW',    (0, 1), (-1, -1), 0.5, C_BORDER),
    ])
    for i in range(1, len(data_rows)):
        if i % 2 == 0:
            style.add('BACKGROUND', (0, i), (-1, i), C_TABLE_ALT)
            
    for r_idx, row in enumerate(data_rows):
        for c_idx, cell in enumerate(row):
            if isinstance(cell, str):
                c = cell.upper()
                if 'HIGH' in c or 'IMMEDIATE' in c or 'SOON' in c:
                    style.add('TEXTCOLOR', (c_idx, r_idx), (c_idx, r_idx), C_ACCENT)
                    style.add('FONTNAME', (c_idx, r_idx), (c_idx, r_idx), 'Helvetica-Bold')
                elif 'MOD' in c or 'MED' in c or 'INVESTIGATE' in c:
                    style.add('TEXTCOLOR', (c_idx, r_idx), (c_idx, r_idx), C_WARN)
                elif 'OK' in c or 'LOW' in c or 'ACCEPTABLE' in c or 'NEGLIGIBLE' in c:
                    style.add('TEXTCOLOR', (c_idx, r_idx), (c_idx, r_idx), C_OK)

    t.setStyle(style)
    return t


# =============================================================================
# Main ReportGenerator class
# =============================================================================
class ReportGenerator:

    JOINT_COLS = {
        'Neck': 'Neck_Flexion_deg',
        'Back': 'Trunk_Flexion_deg',
        'R_Shoulder': 'R_Shoulder_Flexion_deg',
        'L_Shoulder': 'L_Shoulder_Flexion_deg',
        'R_Elbow': 'R_Elbow_Flexion_deg',
        'L_Elbow': 'L_Elbow_Flexion_deg',
        'R_Wrist': 'R_Wrist_Flexion_deg',
        'L_Wrist': 'L_Wrist_Flexion_deg',
        'R_Thigh': 'R_Thigh_Flexion_deg',
        'L_Thigh': 'L_Thigh_Flexion_deg',
        'R_Knee': 'R_Knee_Flexion_deg',
        'L_Knee': 'L_Knee_Flexion_deg'
    }

    def __init__(self, config):
        self.config    = config
        self.ai_models = None
        self._styles   = _build_styles()

    def _load_ai_models(self):
        if not _AI_AVAILABLE:
            return
        try:
            self.ai_models = AIModels(model_dir='models')
            if not self.ai_models.ready:
                self.ai_models = None
        except Exception as e:
            print(f'[ReportGen] AI model load skipped: {e}')
            self.ai_models = None

    def _compute_ai_predictions(self, df):
        """Compute AI predictions (risk_10d, critical joint) for the DataFrame.

        Uses AIModels.predict() which accepts a single feature dict per frame.
        The predictions are computed row-by-row over the whole session.
        """
        if self.ai_models is None:
            return None

        from collections import deque
        from feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor({'feature_cols': self.ai_models.feature_cols})

        risk_10d_list, anomaly_score_list, joint_list, anomaly_probs_list = [], [], [], []
        
        angle_w = deque(maxlen=60)
        risk_w = deque(maxlen=60)
        rula_l_w = deque(maxlen=60)
        rula_r_w = deque(maxlen=60)
        reba_l_w = deque(maxlen=60)
        reba_r_w = deque(maxlen=60)

        condition_list, severity_list, risk_level_list = [], [], []

        for _, row in df.iterrows():
            angle_w.append({c: row[c] for c in df.columns if 'deg' in c or 'Pitch' in c or 'Roll' in c or 'Yaw' in c})
            
            rula_r = row.get('RULA_R_Final', 0.0)
            rula_l = row.get('RULA_L_Final', 0.0)
            reba_r = row.get('REBA_R_Final', 0.0)
            reba_l = row.get('REBA_L_Final', 0.0)
            
            risk_w.append(max(rula_r, rula_l, 0) / 7.0 * 0.5 + max(reba_r, reba_l, 0) / 11.0 * 0.5) 
            rula_l_w.append(rula_l)
            rula_r_w.append(rula_r)
            reba_l_w.append(reba_l)
            reba_r_w.append(reba_r)

            if len(angle_w) >= 60:
                current_time = row['Timestamp'].timestamp() if 'Timestamp' in df.columns and pd.notna(row['Timestamp']) else 0
                f_dict = extractor.extract(angle_w, risk_w, rula_l_w, rula_r_w, reba_l_w, reba_r_w, current_time)
                try:
                    result = self.ai_models.predict(f_dict)
                    risk_10d_list.append(round(float(result.get('risk_10d', 0.0)), 4))
                    anomaly_score_list.append(round(float(result.get('anomaly_score', 0.0)), 4))
                    joint_list.append(result.get('critical_joint') or '')
                    anomaly_probs_list.append(result.get('anomaly_probs', {}))
                    condition_list.append(result.get('condition', 'unknown'))
                    severity_list.append(result.get('severity', 'low'))
                    risk_level_list.append(result.get('risk_level', 'SAFE'))
                except Exception as e:
                    print(f'[AI] prediction error: {e}')
                    risk_10d_list.append(np.nan)
                    anomaly_score_list.append(np.nan)
                    joint_list.append('')
                    anomaly_probs_list.append({})
                    condition_list.append('unknown')
                    severity_list.append('low')
                    risk_level_list.append('SAFE')
            else:
                risk_10d_list.append(np.nan)
                anomaly_score_list.append(np.nan)
                joint_list.append('')
                anomaly_probs_list.append({})
                condition_list.append('unknown')
                severity_list.append('low')
                risk_level_list.append('SAFE')

        return pd.DataFrame({
            'AI_Risk_10d': risk_10d_list,
            'AI_Anomaly_Score': anomaly_score_list,
            'AI_Critical_Joint': joint_list,
            'AI_Anomaly_Probs': anomaly_probs_list,
            'AI_Condition': condition_list,
            'AI_Severity': severity_list,
            'AI_Risk_Level': risk_level_list
        }, index=df.index)

    def _compute_statistics(self, df, ai_df):
        stats = {}

        # Joint angles
        stats['angles'] = {}
        for short_name, col in self.JOINT_COLS.items():
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    stats['angles'][short_name] = dict(
                        min=s.min(), max=s.max(), mean=s.mean(),
                        std=s.std(), p95=s.quantile(0.95)
                    )

        # RULA / REBA
        rula_reba_map = {
            'RULA_R_Final': 'RULA_Right',
            'RULA_L_Final': 'RULA_Left',
            'REBA_R_Final': 'REBA_Right',
            'REBA_L_Final': 'REBA_Left'
        }
        for csv_key, report_key in rula_reba_map.items():
            if csv_key in df.columns:
                s = df[csv_key].dropna()
                if not s.empty:
                    stats[report_key] = dict(
                        min=s.min(), max=s.max(), mean=s.mean(),
                        std=s.std(), p95=s.quantile(0.95),
                        mode=s.mode().iloc[0] if not s.mode().empty else None
                    )

        # AI predictions
        if ai_df is not None:
            risk_10d = ai_df['AI_Risk_10d'].dropna()
            if not risk_10d.empty:
                stats['ai_risk_10d'] = dict(
                    min=risk_10d.min(), max=risk_10d.max(), mean=risk_10d.mean(),
                    std=risk_10d.std(), p95=risk_10d.quantile(0.95),
                    time_high=(risk_10d > 0.8).sum() * 0.1
                )
            
            anomaly_score = ai_df['AI_Anomaly_Score'].dropna()
            if not anomaly_score.empty:
                stats['ai_anomaly'] = dict(
                    min=anomaly_score.min(), max=anomaly_score.max(), mean=anomaly_score.mean(),
                    std=anomaly_score.std(), p95=anomaly_score.quantile(0.95),
                    time_high=(anomaly_score > 0.7).sum() * 0.1
                )

            joints = ai_df['AI_Critical_Joint'].replace('', np.nan).dropna()
            if not joints.empty:
                stats['ai_critical_joint'] = Counter(joints).most_common(5)
                
            # Aggregate condition, severity, risk level
            conditions = ai_df['AI_Condition'].replace('unknown', np.nan).dropna()
            if not conditions.empty:
                stats['ai_condition'] = Counter(conditions).most_common(1)[0][0]
                
            severities = ai_df['AI_Severity'].replace('low', np.nan).dropna()
            if not severities.empty:
                stats['ai_severity'] = Counter(severities).most_common(1)[0][0]
                
            risk_levels = ai_df['AI_Risk_Level'].replace('SAFE', np.nan).dropna()
            if not risk_levels.empty:
                stats['ai_risk_level'] = Counter(risk_levels).most_common(1)[0][0]
                
            # Aggregate anomaly probabilities
            top_anomalies = Counter()
            for probs in ai_df['AI_Anomaly_Probs'].dropna():
                if probs:
                    top_anomaly = max(probs, key=probs.get)
                    if probs[top_anomaly] > 0.5:
                        top_anomalies[top_anomaly] += 1
            if top_anomalies:
                stats['ai_top_anomalies'] = top_anomalies.most_common(5)

        return stats

    def generate(self, csv_file):
        os.makedirs(self.config.REPORTS_DIR, exist_ok=True)

        df = pd.read_csv(csv_file)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        self._load_ai_models()
        ai_df = self._compute_ai_predictions(df) if self.ai_models else None
        stats = self._compute_statistics(df, ai_df)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = os.path.join(self.config.REPORTS_DIR, f'report_{stamp}.pdf')

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2.2 * cm, bottomMargin=1.5 * cm,
            title='Ergo Sensor Risk Report',
            author='Ergo Sensor System',
        )

        story = []
        story += self._section_cover(df, stats)
        story.append(PageBreak())
        story += self._section_executive_summary(stats)
        story.append(PageBreak())
        story += self._section_joint_angles(stats)
        story += self._section_risk_scores(stats)
        story += self._section_ai_insights(stats, ai_df)
        story.append(PageBreak())
        story += self._section_charts(df, ai_df, stats)
        story += self._section_clinical_recommendations(stats)

        doc.build(story,
                  onFirstPage=_on_first_page,
                  onLaterPages=_on_later_pages)

        print(f'[ReportGen] PDF saved -> {pdf_path}')
        return pdf_path

    # =========================================================================
    # Section builders
    # =========================================================================

    def _section_cover(self, df, stats):
        S = self._styles
        story = [Spacer(1, 4 * cm)]
        
        # Cyber title
        story.append(Paragraph("<b>ERGO SENSOR</b>", ParagraphStyle(
            'CoverTitle', parent=self._styles['title'], fontSize=38, textColor=C_NAVY, spaceAfter=2, alignment=TA_LEFT
        )))
        story.append(Paragraph("Musculoskeletal Risk Assessment", ParagraphStyle(
            'CoverSub', parent=self._styles['subtitle'], fontSize=16, textColor=C_TEAL, spaceAfter=0.8*cm, alignment=TA_LEFT
        )))
        
        # Divider
        story.append(HRFlowable(width='100%', thickness=2, color=C_TEAL, spaceBefore=0, spaceAfter=1*cm, hAlign='LEFT'))
        
        # Data period
        if 'Timestamp' in df.columns and df['Timestamp'].notna().any():
            t_min = df['Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            t_max = df['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            duration_s = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
            duration_str = f'{int(duration_s // 60)} min {int(duration_s % 60)} sec'
        else:
            t_min = t_max = 'N/A'
            duration_str = 'N/A'

        meta = [
            ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Session Start',    t_min],
            ['Session End',      t_max],
            ['Session Duration', duration_str],
            ['Total Samples',    f'{len(df):,}'],
            ['Report Type',      'Comprehensive Ergonomic & AI Risk Assessment'],
            ['Classification',   'CONFIDENTIAL — For Medical Use Only'],
        ]
        t = Table(meta, colWidths=[5 * cm, 10 * cm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('FONTNAME',     (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME',     (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE',     (0, 0), (-1, -1), 10),
            ('TEXTCOLOR',    (0, 0), (0, -1), C_NAVY),
            ('TEXTCOLOR',    (1, 0), (1, -1), C_GREY_DARK),
            ('ALIGN',        (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
            ('TOPPADDING',   (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        
        story.append(Spacer(1, 3 * cm))

        # Overall risk badge
        rula_means = [stats.get('RULA_Right', {}).get('mean', 0), stats.get('RULA_Left', {}).get('mean', 0)]
        reba_means = [stats.get('REBA_Right', {}).get('mean', 0), stats.get('REBA_Left', {}).get('mean', 0)]
        max_rula = max(rula_means) if rula_means else 0
        max_reba = max(reba_means) if reba_means else 0
        if max_rula >= 7 or max_reba >= 11:
            badge_text = 'HIGH RISK'
            badge_color = C_ACCENT
        elif max_rula >= 4 or max_reba >= 8:
            badge_text = 'MODERATE RISK'
            badge_color = C_WARN
        else:
            badge_text = 'LOW RISK'
            badge_color = C_OK
            
        story.append(Paragraph("OVERALL RISK PROFILE", ParagraphStyle('RiskHdr', parent=self._styles['subtitle'], alignment=TA_LEFT, textColor=C_GREY_MID)))
        story.append(Paragraph(f"<b>{badge_text}</b>", ParagraphStyle(
            'RiskRating', parent=self._styles['title'], fontSize=32, textColor=badge_color, spaceAfter=2*cm, alignment=TA_LEFT
        )))
        
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            'Generated by automated kinematic monitoring. Requires medical review. '
            'It combines real-time joint angle measurements, RULA/REBA ergonomic scoring, '
            'and LightGBM / Isolation Forest AI predictive models to provide a comprehensive assessment.',
            ParagraphStyle('Disclaimer', parent=self._styles['small'], alignment=TA_LEFT, textColor=C_GREY_MID)
        ))
        return story

    def _section_executive_summary(self, stats):
        S = self._styles
        story = [SectionHeader('EXECUTIVE SUMMARY', bg=C_NAVY), Spacer(1, 0.4 * cm)]

        rows = []
        # RULA / REBA means
        for key in ['RULA_Right', 'RULA_Left', 'REBA_Right', 'REBA_Left']:
            if key in stats:
                fn = _rula_colour if 'RULA' in key else _reba_colour
                rows.append((f'{key.replace("_", " ")} — Mean',
                             f"{stats[key]['mean']:.1f}", fn(stats[key]['mean'])))
                rows.append((f'{key.replace("_", " ")} — Peak',
                             f"{stats[key]['max']:.0f}", fn(stats[key]['max'])))

        # AI
        if 'ai_risk_10d' in stats:
            p = stats['ai_risk_10d']
            rows.append(('AI 10-day Risk Forecast — Mean', f"{p['mean']:.3f}", _rula_colour(p['mean'] * 10)))
            rows.append(('AI 10-day Risk Forecast — Peak', f"{p['max']:.3f}", _rula_colour(p['max'] * 10)))
            rows.append(('AI High-Risk Duration', f"{p['time_high']:.0f} s", C_ACCENT if p['time_high'] > 0 else C_OK))
        if 'ai_condition' in stats and stats['ai_condition'] != 'normal':
            rows.append(('AI Predicted Condition', str(stats['ai_condition']).replace('_', ' ').title(), C_WARN))
        if 'ai_severity' in stats and stats['ai_severity'] != 'low':
            sev = stats['ai_severity']
            col = C_ACCENT if sev == 'high' else C_WARN
            rows.append(('AI Predicted Severity', str(sev).upper(), col))
        if 'ai_anomaly' in stats:
            a = stats['ai_anomaly']
            rows.append(('AI Anomaly Score — Peak', f"{a['max']:.3f}", _rula_colour(a['max'] * 10)))
        if 'ai_critical_joint' in stats:
            top_joint = stats['ai_critical_joint'][0][0]
            rows.append(('Most Critical Joint (AI)', top_joint, C_NAVY))

        # Build table
        table_rows = [['Metric', 'Value', 'Flag']]
        for i, (label, value, col) in enumerate(rows, start=1):
            flag = 'HIGH' if col == C_ACCENT else ('MOD' if col == C_WARN else 'OK')
            table_rows.append([label, value, flag])

        t = Table(table_rows, colWidths=[8 * cm, 4 * cm, 2.5 * cm])
        ts = [
            ('BACKGROUND',   (0, 0), (-1, 0), C_NAVY),
            ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
            ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0, 0), (-1, -1), 8),
            ('ALIGN',        (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_WHITE, C_GREY_LITE]),
            ('GRID',         (0, 0), (-1, -1), 0.4, C_BORDER),
            ('TOPPADDING',   (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ]
        for i, (_, _, col) in enumerate(rows, start=1):
            ts.append(('TEXTCOLOR', (2, i), (2, i), col))
            ts.append(('FONTNAME',  (2, i), (2, i), 'Helvetica-Bold'))
        t.setStyle(TableStyle(ts))
        story.append(t)
        return story

    def _section_joint_angles(self, stats):
        S = self._styles
        story = [Spacer(1, 0.5 * cm),
                 SectionHeader('1. JOINT ANGLE STATISTICS (degrees)', bg=C_TEAL),
                 Spacer(1, 0.3 * cm)]

        if not stats.get('angles'):
            story.append(Paragraph('No joint angle data available.', S['body']))
            return story

        hdr = ['Joint', 'Min', 'Max', 'Mean', 'Std Dev', '95th %ile']
        rows = [hdr]
        for joint, v in stats['angles'].items():
            rows.append([
                joint,
                f"{v['min']:.1f}",
                f"{v['max']:.1f}",
                f"{v['mean']:.1f}",
                f"{v['std']:.1f}",
                f"{v['p95']:.1f}",
            ])
        story.append(_styled_table(rows, [3.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.5*cm]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            'Reference: Neck flexion >20 deg, shoulder elevation >60 deg, '
            'elbow flexion outside 60-100 deg range, or wrist deviation >15 deg '
            'are associated with elevated MSD risk.',
            S['small']
        ))
        return story

    def _section_risk_scores(self, stats):
        S = self._styles
        story = [Spacer(1, 0.5 * cm),
                 SectionHeader('2. ERGONOMIC RISK SCORES (RULA / REBA)', bg=C_TEAL),
                 Spacer(1, 0.3 * cm)]

        score_keys = [k for k in ['RULA_Right','RULA_Left','REBA_Right','REBA_Left']
                      if k in stats]
        if score_keys:
            hdr  = ['Score', 'Min', 'Max', 'Mean', 'Std', '95th %ile', 'Most Frequent', 'Action Level']
            rows = [hdr]
            for key in score_keys:
                v   = stats[key]
                alv = self._action_level(key, v['mean'])
                rows.append([
                    key.replace('_', ' '),
                    f"{v['min']:.0f}",
                    f"{v['max']:.0f}",
                    f"{v['mean']:.1f}",
                    f"{v['std']:.1f}",
                    f"{v['p95']:.0f}",
                    str(int(v['mode'])) if v['mode'] is not None else '-',
                    alv,
                ])
            t = _styled_table(rows, [2.8*cm,1.4*cm,1.4*cm,1.6*cm,1.4*cm,1.8*cm,2*cm,2.6*cm])
            story.append(t)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                'RULA Action: 1-2 Acceptable | 3-4 Further investigation | '
                '5-6 Change soon | 7+ Change immediately.  '
                'REBA Action: 1 Negligible | 2-3 Low | 4-7 Medium | 8-10 High | 11+ Very High.',
                S['small']
            ))
        else:
            story.append(Paragraph('RULA/REBA data not available in this session.', S['body']))

        return story

    def _action_level(self, key, mean_val):
        if 'RULA' in key:
            if mean_val >= 7: return 'IMMEDIATE'
            if mean_val >= 5: return 'Soon'
            if mean_val >= 3: return 'Investigate'
            return 'Acceptable'
        else:
            if mean_val >= 11: return 'VERY HIGH'
            if mean_val >= 8:  return 'HIGH'
            if mean_val >= 4:  return 'MEDIUM'
            if mean_val >= 2:  return 'LOW'
            return 'Negligible'

    def _section_ai_insights(self, stats, ai_df):
        S = self._styles
        story = [Spacer(1, 0.5 * cm),
                 SectionHeader('3. AI PREDICTIVE INSIGHTS (LightGBM & Isolation Forest)', bg=C_NAVY),
                 Spacer(1, 0.3 * cm)]

        if 'ai_risk_10d' not in stats and 'ai_critical_joint' not in stats:
            story.append(Paragraph(
                'AI predictions were not available for this session. '
                'This may be because the session was shorter than 60 frames '
                '(the minimum required for the sliding window), or the AI models '
                'could not be loaded.',
                S['body']
            ))
            return story

        if 'ai_risk_10d' in stats:
            p = stats['ai_risk_10d']
            a = stats.get('ai_anomaly', {})
            story.append(Paragraph('<b>AI Risk, Condition & Anomaly Overview</b>', S['bold']))
            story.append(Spacer(1, 0.15 * cm))

            rows = [['Metric', '10-day Risk', 'Anomaly (IsoForest)']]
            rows.append(['Mean Score', f"{p['mean']:.3f}", f"{a.get('mean', 0.0):.3f}"])
            rows.append(['Peak Score', f"{p['max']:.3f}", f"{a.get('max', 0.0):.3f}"])
            
            if 'ai_condition' in stats:
                rows.append(['Predicted Condition', str(stats['ai_condition']).replace('_', ' ').title(), ''])
            if 'ai_severity' in stats:
                rows.append(['Severity', str(stats['ai_severity']).upper(), ''])
            if 'ai_risk_level' in stats:
                rows.append(['Overall Risk Level', str(stats['ai_risk_level']).upper(), ''])

            t = _styled_table(rows, [4 * cm, 5 * cm, 5 * cm])
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))

        if 'ai_top_anomalies' in stats:
            story.append(Paragraph('<b>Detected Postural Anomalies</b>', S['bold']))
            story.append(Spacer(1, 0.15 * cm))
            rows = [['Rank', 'Anomaly Type', 'Frames Detected']]
            for rank, (anomaly, count) in enumerate(stats['ai_top_anomalies'], 1):
                rows.append([str(rank), anomaly, str(count)])
            t = _styled_table(rows, [2 * cm, 8 * cm, 4 * cm])
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))

        if 'ai_critical_joint' in stats:
            story.append(Paragraph('<b>Most Frequently Critical Joints (SHAP Analysis)</b>', S['bold']))
            story.append(Spacer(1, 0.15 * cm))
            rows = [['Rank', 'Joint', 'Occurrences', 'Clinical Relevance']]
            for rank, (joint, count) in enumerate(stats['ai_critical_joint'], 1):
                rows.append([str(rank), joint, str(count), self._joint_advice(joint)])
            t = _styled_table(rows, [1.5*cm, 3*cm, 3*cm, 7*cm])
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))

        story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER))
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(
            '<b>Methodology:</b> The LightGBM classifier analyses the last 60 frames '
            '(~6 seconds at 10 Hz) of joint angle statistics to forecast risk exceedance '
            'over the next 10 days. An Isolation Forest model identifies anomalous movements, '
            'and specific LightGBM anomaly models flag granular postural deviations. '
            'SHAP TreeExplainer identifies which joints contribute most to the risk prediction.',
            S['small']
        ))
        return story

    def _interp_prob(self, val):
        if val >= 0.7: return 'HIGH — Immediate attention required'
        if val >= 0.5: return 'MODERATE — Monitor closely'
        return 'LOW — Acceptable range'

    def _joint_advice(self, joint):
        advice = {
            'Neck':       'Check head/neck posture; consider ergonomic support',
            'Back':       'Review trunk flexion; lumbar support recommended',
            'R_Shoulder': 'Reduce right arm elevation; check workstation height',
            'L_Shoulder': 'Reduce left arm elevation; check workstation height',
            'R_Elbow':    'Adjust reach distance; maintain elbow 90-100 deg',
            'L_Elbow':    'Adjust reach distance; maintain elbow 90-100 deg',
            'R_Wrist':    'Minimise right wrist deviation; use neutral grip tools',
            'L_Wrist':    'Minimise left wrist deviation; use neutral grip tools',
        }
        return advice.get(joint, 'Review joint posture during task')

    def _section_charts(self, df, ai_df, stats):
        S = self._styles
        story = [SectionHeader('4. TREND CHARTS', bg=C_NAVY), Spacer(1, 0.3 * cm)]

        def _style_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#E2E8F0')
            ax.spines['bottom'].set_color('#E2E8F0')
            ax.grid(True, linestyle='--', color='#F1F5F9', alpha=0.8)
            ax.tick_params(axis='x', colors='#64748B', labelsize=7)
            ax.tick_params(axis='y', colors='#64748B', labelsize=7)
            ax.yaxis.label.set_color('#64748B')
            ax.xaxis.label.set_color('#64748B')
            ax.title.set_color('#0F172A')
            ax.title.set_fontsize(10)
            ax.title.set_fontweight('bold')

        # 1. AI 10-day Risk & Anomaly Score
        if ai_df is not None and 'AI_Risk_10d' in ai_df.columns:
            risk_series = ai_df['AI_Risk_10d'].dropna()
            anomaly_series = ai_df.get('AI_Anomaly_Score')
            if not risk_series.empty and 'Timestamp' in df.columns:
                valid_idx = risk_series.index
                fig, ax = plt.subplots(figsize=(9, 3.5))
                
                # Plot Risk 10d
                ax.plot(df.loc[valid_idx, 'Timestamp'], risk_series,
                        lw=1.2, color='#E84545', label='10-day Risk Forecast')
                
                # Plot Anomaly if available
                if anomaly_series is not None and not anomaly_series.dropna().empty:
                    valid_idx_a = anomaly_series.dropna().index
                    ax.plot(df.loc[valid_idx_a, 'Timestamp'], anomaly_series.dropna(),
                            lw=0.8, color='#0D7377', linestyle='--', label='Anomaly Score')
                
                ax.axhline(0.8, color='#E84545', ls=':', lw=1, alpha=0.5, label='High Risk (0.8)')
                ax.fill_between(df.loc[valid_idx, 'Timestamp'], risk_series, 0,
                                where=risk_series >= 0.8,
                                alpha=0.1, color='#E84545')
                                
                ax.set_title('AI Postural Risk & Anomaly Tracking')
                ax.set_xlabel('Time')
                ax.set_ylabel('Score (0.0 to 1.0)')
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7, loc='upper left')
                _style_ax(ax)
                fig.tight_layout()
                story.append(Paragraph('<b>AI Risk Tracking Trends</b>', S['bold']))
                story.append(_fig_to_image(fig, height=6 * cm))
                story.append(Spacer(1, 0.3 * cm))


        # 2. Joint Angles
        available_joints = [name for name, c in self.JOINT_COLS.items() if c in df.columns]
        if available_joints and 'Timestamp' in df.columns:
            colours_map = {
                'Neck': '#1A3A5C', 'Back': '#0D7377',
                'R_Shoulder': '#E84545', 'L_Shoulder': '#F5A623',
                'R_Elbow': '#27AE60', 'L_Elbow': '#8E44AD',
                'R_Wrist': '#2980B9', 'L_Wrist': '#E67E22',
                'R_Thigh': '#16A085', 'L_Thigh': '#C0392B',
                'R_Knee': '#2C3E50', 'L_Knee': '#D35400'
            }
            # Adjust grid size based on number of plots. We have up to 12 joints now!
            n_cols = 4
            n_rows = (len(self.JOINT_COLS) + 3) // 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), sharex=True)
            axes = axes.flatten()
            for idx, (joint_name, col) in enumerate(self.JOINT_COLS.items()):
                ax = axes[idx]
                if col in df.columns:
                    ax.plot(df['Timestamp'], df[col],
                            lw=0.7, color=colours_map.get(joint_name, 'grey'))
                    ax.set_title(joint_name)
                    _style_ax(ax)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, color='#E2E8F0')
                    ax.set_title(joint_name)
                    _style_ax(ax)
            # hide any remaining axes
            for idx in range(len(self.JOINT_COLS), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle('All Joint Angles Over Time (degrees)',
                         fontsize=10, fontweight='bold', y=1.01)
            fig.tight_layout()
            story.append(PageBreak())
            story.append(Paragraph('<b>Joint Angle Trends</b>', S['bold']))
            story.append(_fig_to_image(fig, width=16 * cm, height=9 * cm))
            story.append(Spacer(1, 0.3 * cm))

        # 3. RULA scores
        rula_cols = [c for c in ['RULA_R_Final', 'RULA_L_Final'] if c in df.columns]
        if rula_cols and 'Timestamp' in df.columns:
            fig, ax = plt.subplots(figsize=(9, 3))
            for col, clr in zip(rula_cols, ['#E84545', '#2980B9']):
                label = 'RULA Right' if 'R_Final' in col else 'RULA Left'
                ax.plot(df['Timestamp'], df[col], lw=0.8, color=clr, label=label)
            for thresh, lbl, clr in [(7,'Immediate','#FF4D6D'),(5,'Soon','#FFAA00'),(3,'Investigate','#00E5A0')]:
                ax.axhline(thresh, color=clr, ls=':', lw=0.8, alpha=0.7, label=f'RULA {lbl}')
            ax.set_title('RULA Scores Over Time')
            ax.set_ylabel('RULA Score')
            ax.legend(fontsize=7, ncol=3)
            ax.set_ylim(1, 7.5)
            _style_ax(ax)
            fig.tight_layout()
            story.append(Paragraph('<b>RULA Ergonomic Scores</b>', S['bold']))
            story.append(_fig_to_image(fig, height=5.5 * cm))
            story.append(Spacer(1, 0.3 * cm))

        # 4. REBA scores
        reba_cols = [c for c in ['REBA_R_Final', 'REBA_L_Final'] if c in df.columns]
        if reba_cols and 'Timestamp' in df.columns:
            fig, ax = plt.subplots(figsize=(9, 3))
            for col, clr in zip(reba_cols, ['#E84545', '#2980B9']):
                label = 'REBA Right' if 'R_Final' in col else 'REBA Left'
                ax.plot(df['Timestamp'], df[col], lw=0.8, color=clr, label=label)
            for thresh, lbl, clr in [(11,'Very High','#FF4D6D'),(8,'High','#FFAA00'),(4,'Medium','#FFAA00')]:
                ax.axhline(thresh, color=clr, ls=':', lw=0.8, alpha=0.7, label=f'REBA {lbl}')
            ax.set_title('REBA Scores Over Time')
            ax.set_ylabel('REBA Score')
            ax.legend(fontsize=7, ncol=3)
            ax.set_ylim(1, 15.5)
            _style_ax(ax)
            fig.tight_layout()
            story.append(Paragraph('<b>REBA Ergonomic Scores</b>', S['bold']))
            story.append(_fig_to_image(fig, height=5.5 * cm))
            story.append(Spacer(1, 0.3 * cm))

        # 5. Critical joints bar chart
        if 'ai_critical_joint' in stats:
            joints  = [j for j, _ in stats['ai_critical_joint']]
            counts  = [c for _, c in stats['ai_critical_joint']]
            colours = ['#FF4D6D', '#FFAA00', '#00E5A0', '#00E5FF', '#8E44AD']
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(joints[::-1], counts[::-1],
                           color=colours[:len(joints)], edgecolor='white')
            ax.set_title('Most Critical Joints (AI SHAP Analysis)')
            ax.set_xlabel('Occurrences')
            _style_ax(ax)
            for bar, val in zip(bars, counts[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        str(val), va='center', fontsize=8, color='#64748B')
            fig.tight_layout()
            story.append(Paragraph('<b>AI-Identified Critical Joints</b>', S['bold']))
            story.append(_fig_to_image(fig, width=10 * cm, height=5 * cm))

        return story

    def _section_clinical_recommendations(self, stats):
        S = self._styles
        story = [PageBreak(),
                 SectionHeader('5. CLINICAL RECOMMENDATIONS', bg=C_ACCENT),
                 Spacer(1, 0.4 * cm)]

        recs = []

        # RULA/REBA
        for key in ['RULA_Right', 'RULA_Left']:
            if key in stats and stats[key]['mean'] >= 5:
                side = 'right' if 'Right' in key else 'left'
                recs.append(('RULA',
                             f'{side.capitalize()} side RULA score averages {stats[key]["mean"]:.1f}. '
                             f'Reduce {side} arm reach, adjust elbow height, and minimise wrist deviation.'))

        for key in ['REBA_Right', 'REBA_Left']:
            if key in stats and stats[key]['mean'] >= 8:
                side = 'right' if 'Right' in key else 'left'
                recs.append(('REBA',
                             f'{side.capitalize()} REBA score is HIGH (avg {stats[key]["mean"]:.1f}). '
                             'Whole-body posture review required; reduce trunk and neck flexion.'))

        # AI probability
        if 'ai_risk_10d' in stats and stats['ai_risk_10d']['mean'] >= 0.5:
            recs.append(('AI ALERT',
                         f"LightGBM model forecasts elevated 10-day risk (average score "
                         f"{stats['ai_risk_10d']['mean']:.2f}). "
                         'Frequent posture corrections and mandatory micro-breaks every 20 minutes.'))
        if 'ai_condition' in stats and stats['ai_condition'] != 'normal':
            condition_str = str(stats['ai_condition']).replace('_', ' ').title()
            severity_str = str(stats.get('ai_severity', 'low')).upper()
            recs.append(('AI MEDICAL',
                         f"AI model predicted a {severity_str} severity condition: {condition_str}. "
                         'Recommend immediate medical review or ergonomic intervention.'))
        if 'ai_anomaly' in stats and stats['ai_anomaly']['max'] >= 0.7:
            recs.append(('AI ALERT',
                         f"Isolation Forest identified critical postural anomalies. "
                         'Immediate technique correction and ergonomic evaluation advised.'))

        # Critical joint
        if 'ai_critical_joint' in stats:
            top = stats['ai_critical_joint'][0][0]
            recs.append(('JOINT FOCUS',
                         f'The AI model identified <b>{top}</b> as the most critical joint. '
                         f'{self._joint_advice(top)}.'))

        # Draw recommendations table
        rows = [['Priority', 'Recommendation']]
        color_map = {
            'AI ALERT': C_ACCENT,
            'AI MEDICAL': C_ACCENT,
            'RULA': C_WARN, 'REBA': C_WARN,
            'JOINT FOCUS': C_TEAL,
        }
        for tag, text in recs:
            rows.append([tag, Paragraph(text, S['body'])])

        if len(rows) == 1:
            rows.append(['NO ACTION', 'No significant risks detected. Continue periodic monitoring.'])

        t = Table(rows, colWidths=[3 * cm, 11.5 * cm])
        ts_rules = [
            ('BACKGROUND',  (0, 0), (-1, 0), C_ACCENT),
            ('TEXTCOLOR',   (0, 0), (-1, 0), C_WHITE),
            ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, 0), 9),
            ('ALIGN',       (0, 0), (0, -1), 'CENTER'),
            ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME',    (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 1), (0, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_WHITE, C_GREY_LITE]),
            ('GRID',        (0, 0), (-1, -1), 0.4, C_BORDER),
            ('TOPPADDING',  (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]
        for i, (tag, _) in enumerate(recs, start=1):
            col = color_map.get(tag, C_GREY_MID)
            ts_rules.append(('TEXTCOLOR', (0, i), (0, i), col))
        t.setStyle(TableStyle(ts_rules))
        story.append(t)

        # Disclaimer
        story.append(Spacer(1, 0.6 * cm))
        story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            '<b>Disclaimer:</b> This report is generated automatically by the Ergo Sensor system '
            'and is intended as a decision-support tool only. It does not replace professional '
            'medical or occupational health evaluation. All recommendations should be reviewed '
            'and validated by a qualified healthcare professional before implementation.',
            S['small']
        ))
        return story