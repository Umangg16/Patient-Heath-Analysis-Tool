import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from recommendation_engine import ComprehensiveRecommendationSystem

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Patient Health Analysis & Wellness Guidance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #2874A6;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2874A6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Danger box */
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'report' not in st.session_state:
    st.session_state.report = None

if 'system' not in st.session_state:
    with st.spinner("🔄 Initializing recommendation system..."):
        try:
            st.session_state.system = ComprehensiveRecommendationSystem(
                model_dir='models',
                profiles_path='data/profiles/comprehensive_disease_profiles.json',
                use_llm=False  # Set to True if Ollama is running
            )
            st.session_state.system_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            st.session_state.system_loaded = False

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        🏥 Patient Health Analysis & Wellness Guidance Tool
    </div>
    """, unsafe_allow_html=True)
    
    # Subtitle with model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔬 MIMIC-IV Disease Prediction**")
        st.caption("Random Forest (F1: 87.8%, AUROC: 98.5%)")
    with col2:
        st.markdown("**📊 NHANES 2021-2023 Population Data**")
        st.caption("11,933 participants analyzed")
    with col3:
        st.markdown("**🤖 AI-Powered Recommendations**")
        st.caption("Personalized lifestyle guidance")
    
    st.markdown("---")
    
    # Check if system loaded
    if not st.session_state.system_loaded:
        st.error("❌ System failed to load. Please check model files in 'models/' directory.")
        return
    
    # ========================================================================
    # SIDEBAR - PATIENT INPUT
    # ========================================================================
    
    with st.sidebar:
        st.header("📋 Patient Information")
        
        # Instructions
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            **Step 1:** Enter patient demographics  
            **Step 2:** Describe symptoms in natural language  
            **Step 3:** Add available lab values (optional)  
            **Step 4:** Click "🔬 Analyze Health" button  
            
            The system will:
            - Predict potential diseases
            - Compare to population data
            - Generate personalized recommendations
            """)
        
        st.markdown("---")
        
        # Demographics
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=45, help="Patient's age in years")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # Convert gender to M/F for model
        gender_code = 'M' if gender == 'Male' else 'F'
        
        # BMI calculation helper
        st.markdown("**BMI Calculator**")
        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
        
        bmi = round(weight_kg / ((height_cm/100) ** 2), 1)
        st.info(f"📊 Calculated BMI: **{bmi}** kg/m²")
        
        st.markdown("---")
        
        # Symptoms
        st.subheader("🩺 Symptoms")
        symptoms_input_method = st.radio(
            "Input method:",
            ["Text description", "Select from list"],
            help="Choose how to enter symptoms"
        )
        
        if symptoms_input_method == "Text description":
            symptoms_text = st.text_area(
                "Describe symptoms:",
                placeholder="Example: I have increased thirst, frequent urination, fatigue, and blurred vision",
                height=120,
                help="Describe symptoms in natural language"
            )
        else:
            common_symptoms = {
                'Diabetes': ['Increased thirst', 'Frequent urination', 'Fatigue', 'Blurred vision', 'Slow healing wounds'],
                'Hypertension': ['Headache', 'Dizziness', 'Chest pain', 'Difficulty concentrating'],
                'Heart Failure': ['Shortness of breath', 'Fatigue', 'Swelling in legs', 'Rapid heartbeat'],
                'Kidney Failure': ['Fatigue', 'Nausea', 'Decreased urine output', 'Swelling', 'Metallic taste'],
                'Asthma': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Coughing'],
                'Arthritis': ['Joint pain', 'Stiffness', 'Swelling in joints', 'Reduced mobility'],
                'General': ['Fever', 'Weight loss', 'Night sweats', 'Loss of appetite']
            }
            
            selected_category = st.selectbox("Symptom category:", list(common_symptoms.keys()))
            selected_symptoms = st.multiselect(
                "Select symptoms:",
                common_symptoms[selected_category],
                help="Select all that apply"
            )
            symptoms_text = ', '.join(selected_symptoms)
        
        st.markdown("---")
        
        # Clinical Measurements
        st.subheader("🧪 Clinical Measurements")
        st.caption("Optional - provide available lab values")
        
        with st.expander("Blood Work", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                glucose = st.number_input(
                    "Glucose (mg/dL)", 
                    min_value=0, max_value=500, value=100, 
                    help="Fasting glucose level"
                )
                hematocrit = st.number_input(
                    "Hematocrit (%)", 
                    min_value=0.0, max_value=60.0, value=40.0, step=0.1,
                    help="Percentage of red blood cells"
                )
                creatinine = st.number_input(
                    "Creatinine (mg/dL)", 
                    min_value=0.0, max_value=15.0, value=1.0, step=0.1,
                    help="Kidney function marker"
                )
            
            with col2:
                sodium = st.number_input(
                    "Sodium (mEq/L)", 
                    min_value=0, max_value=200, value=140,
                    help="Electrolyte balance"
                )
                potassium = st.number_input(
                    "Potassium (mEq/L)", 
                    min_value=0.0, max_value=10.0, value=4.0, step=0.1,
                    help="Electrolyte balance"
                )
                urea_nitrogen = st.number_input(
                    "BUN (mg/dL)", 
                    min_value=0, max_value=200, value=15,
                    help="Blood Urea Nitrogen"
                )
        
        with st.expander("Blood Pressure", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
            with col2:
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        
        st.markdown("---")
        
        # Analyze button
        analyze_button = st.button("🔬 Analyze Health", type="primary", use_container_width=True)
        
        # Clear history button
        if st.session_state.history:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.report = None
                st.rerun()
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    if analyze_button:
        if not symptoms_text.strip():
            st.warning("⚠️ Please enter symptoms before analyzing.")
        else:
            # Prepare user input
            user_input = {
                'age': age,
                'gender': gender_code,
                'symptoms_text': symptoms_text,
                'bmi': bmi,
                'glucose': glucose if glucose > 0 else None,
                'systolic_bp': systolic_bp if systolic_bp > 0 else None,
                'diastolic_bp': diastolic_bp if diastolic_bp > 0 else None,
                'hematocrit': hematocrit if hematocrit > 0 else None,
                'creatinine': creatinine if creatinine > 0 else None,
                'sodium': sodium if sodium > 0 else None,
                'potassium': potassium if potassium > 0 else None,
                'urea_nitrogen': urea_nitrogen if urea_nitrogen > 0 else None
            }
            
            # Generate report
            with st.spinner("🔄 Analyzing health data... This may take 30-60 seconds..."):
                try:
                    report = st.session_state.system.generate_comprehensive_report(user_input)
                    st.session_state.report = report
                    
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'disease': report['prediction']['disease'],
                        'confidence': report['prediction']['confidence'],
                        'age': age,
                        'gender': gender
                    })
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.exception(e)
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    if st.session_state.report:
        report = st.session_state.report
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔬 Prediction Results", 
            "📊 Population Comparison", 
            "💡 Recommendations",
            "📈 Visualizations",
            "📄 Full Report"
        ])
        
        # TAB 1: PREDICTION RESULTS
        with tab1:
            display_prediction_results(report)
        
        # TAB 2: POPULATION COMPARISON
        with tab2:
            display_population_comparison(report)
        
        # TAB 3: RECOMMENDATIONS
        with tab3:
            display_recommendations(report)
        
        # TAB 4: VISUALIZATIONS
        with tab4:
            display_visualizations(report)
        
        # TAB 5: FULL REPORT
        with tab5:
            display_full_report(report)
    
    else:
        # Welcome message when no analysis yet
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to the Patient Health Analysis System</h3>
            <p>This intelligent system uses:</p>
            <ul>
                <li><strong>MIMIC-IV trained AI model</strong> for accurate disease prediction</li>
                <li><strong>NHANES population data</strong> to compare your health markers</li>
                <li><strong>Evidence-based guidelines</strong> from AHA, ADA, and CDC</li>
            </ul>
            <p>Enter patient information in the sidebar to get started ➡️</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample cases
        st.subheader("📚 Example Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Example 1: Diabetes**
            - Age: 55, Male
            - Symptoms: Increased thirst, frequent urination
            - Glucose: 148 mg/dL
            - BMI: 32.5
            """)
        
        with col2:
            st.markdown("""
            **Example 2: Hypertension**
            - Age: 62, Female
            - Symptoms: Headache, dizziness
            - BP: 152/96 mmHg
            - BMI: 28.5
            """)
        
        with col3:
            st.markdown("""
            **Example 3: Kidney Failure**
            - Age: 68, Male
            - Symptoms: Fatigue, swelling
            - Creatinine: 2.8 mg/dL
            - BUN: 45 mg/dL
            """)
    
    # ========================================================================
    # SIDEBAR - ANALYSIS HISTORY
    # ========================================================================
    
    if st.session_state.history:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📜 Analysis History")
            
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df[['timestamp', 'disease', 'confidence']],
                use_container_width=True,
                hide_index=True
            )

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_prediction_results(report):
    """Display disease prediction results"""
    st.header("🔬 Disease Prediction Results")
    
    prediction = report['prediction']
    
    # Main prediction card
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color: #2874A6;">Predicted Condition: {prediction['disease']}</h2>
        <h3 style="margin:10px 0 0 0; color: #555;">Confidence: {prediction['confidence']*100:.1f}%</h3>
        <p style="margin:5px 0 0 0; color: #777;">Model: {report['metadata']['model_version']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    confidence = prediction['confidence'] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence Level", f"{confidence:.1f}%")
    
    with col2:
        if confidence >= 80:
            st.success("✅ High Confidence")
        elif confidence >= 60:
            st.warning("⚠️ Moderate Confidence")
        else:
            st.info("ℹ️ Low Confidence")
    
    with col3:
        st.metric("Model Performance", "F1: 87.8%")
    
    st.markdown("---")
    
    # Top 3 predictions
    st.subheader("📊 Top 3 Predicted Conditions")
    
    top_3 = prediction['top_3_predictions']
    
    # Create horizontal bar chart
    diseases = list(top_3.keys())
    probabilities = [top_3[d] * 100 for d in diseases]
    
    fig = go.Figure(data=[
        go.Bar(
            y=diseases,
            x=probabilities,
            orientation='h',
            marker=dict(
                color=probabilities,
                colorscale='Blues',
                showscale=False,
                line=dict(color='#2874A6', width=2)
            ),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='outside',
            hovertemplate='%{y}<br>Probability: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=250,
        margin=dict(l=0, r=50, t=40, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(range=[0, max(probabilities) * 1.2])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # All probabilities
    with st.expander("📋 All Disease Probabilities", expanded=False):
        all_probs = prediction['all_probabilities']
        prob_df = pd.DataFrame({
            'Disease': list(all_probs.keys()),
            'Probability': [f"{v*100:.2f}%" for v in all_probs.values()]
        }).sort_values('Probability', ascending=False)
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

def display_population_comparison(report):
    """Display population comparison"""
    st.header("📊 Comparison to NHANES Population")
    
    pop_analysis = report['population_analysis']
    
    # Population info card
    st.markdown(f"""
    <div class="info-box">
        <h4>Reference Population</h4>
        <p><strong>Profile:</strong> {pop_analysis['nhanes_profile']}</p>
        <p><strong>Sample Size:</strong> {pop_analysis['sample_size']:,} participants</p>
        <p><strong>Prevalence:</strong> {pop_analysis['prevalence']}</p>
        <p><strong>Data Source:</strong> NHANES 2021-2023</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clinical markers comparison
    st.subheader("🎯 Your Values vs Population")
    
    comparisons = pop_analysis['comparisons']
    
    if comparisons:
        for marker, data in comparisons.items():
            with st.expander(f"📊 {marker.replace('_', ' ').title()}", expanded=True):
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'user_value' in data:
                        st.metric("Your Value", f"{data['user_value']}")
                    elif 'user_systolic' in data:
                        st.metric("Your Value", f"{data['user_systolic']}/{data['user_diastolic']}")
                
                with col2:
                    st.metric("Population Avg", f"{data.get('population_mean', 'N/A')}")
                
                with col3:
                    st.metric("Target", data['target'])
                
                with col4:
                    # Deviation indicator
                    if 'deviation' in data:
                        deviation = data['deviation']
                        if abs(deviation) < 5:
                            st.metric("Deviation", f"{deviation:+.1f}", delta_color="off")
                        elif deviation > 0:
                            st.metric("Deviation", f"{deviation:+.1f}", delta_color="inverse")
                        else:
                            st.metric("Deviation", f"{deviation:+.1f}", delta_color="normal")
                
                # Status indicator
                status = data['status']
                if status in ['normal', 'desirable']:
                    st.markdown(f'<div class="success-box">✅ Status: <strong>{status.title()}</strong></div>', 
                              unsafe_allow_html=True)
                elif status in ['elevated', 'borderline high', 'overweight', 'prediabetic range']:
                    st.markdown(f'<div class="warning-box">⚠️ Status: <strong>{status.title()}</strong></div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="danger-box">🚨 Status: <strong>{status.title()}</strong></div>', 
                              unsafe_allow_html=True)
                
                # Visual gauge for single values
                if 'user_value' in data and data['user_value']:
                    create_gauge_chart(marker, data)
    
    else:
        st.info("ℹ️ No lab values provided for population comparison. Add lab values in the sidebar to see detailed comparisons.")

def create_gauge_chart(marker, data):
    """Create gauge chart for clinical marker"""
    
    user_val = data.get('user_value', 0)
    
    # Define ranges based on marker
    if 'glucose' in marker.lower():
        ranges = [0, 100, 126, 200, 300]
        colors = ['green', 'yellow', 'orange', 'red']
    elif 'creatinine' in marker.lower():
        ranges = [0, 1.2, 1.5, 3.0, 5.0]
        colors = ['green', 'yellow', 'orange', 'red']
    elif 'bmi' in marker.lower():
        ranges = [0, 18.5, 25, 30, 50]
        colors = ['yellow', 'green', 'orange', 'red']
    else:
        return  # Skip gauge for complex markers
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = user_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': marker.replace('_', ' ').title()},
        gauge = {
            'axis': {'range': [ranges[0], ranges[-1]]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [ranges[i], ranges[i+1]], 'color': colors[i]}
                for i in range(len(colors))
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': user_val
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations(report):
    """Display AI-generated recommendations"""
    st.header("💡 Personalized Health Recommendations")
    
    recommendations = report['recommendations']
    
    # Display recommendations
    st.markdown(recommendations)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Download as Markdown",
            data=recommendations,
            file_name=f"recommendations_{report['prediction']['disease'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        # Convert to PDF would go here
        st.button("📄 Generate PDF", disabled=True, use_container_width=True, 
                 help="PDF generation coming soon")
    
    with col3:
        st.button("📧 Email Report", disabled=True, use_container_width=True,
                 help="Email integration coming soon")

def display_visualizations(report):
    """Display visualizations and charts"""
    st.header("📈 Health Data Visualizations")
    
    # Risk factors radar chart
    st.subheader("🎯 Risk Factor Analysis")
    
    comparisons = report['population_analysis']['comparisons']
    
    if comparisons:
        # Create risk score based on deviations
        categories = []
        values = []
        
        for marker, data in comparisons.items():
            categories.append(marker.replace('_', ' ').title())
            
            # Calculate risk score (0-100)
            status = data['status']
            if status in ['normal', 'desirable']:
                risk = 20
            elif status in ['elevated', 'borderline high', 'overweight', 'prediabetic range']:
                risk = 50
            elif status in ['high', 'obese', 'diabetic range', 'stage 1 hypertension']:
                risk = 75
            else:
                risk = 95
            
            values.append(risk)
        
        # Radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Risk Profile',
            line=dict(color='#dc3545', width=2),
            fillcolor='rgba(220, 53, 69, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[30] * len(categories),  # Healthy baseline
            theta=categories,
            fill='toself',
            name='Healthy Range',
            line=dict(color='#28a745', width=2, dash='dot'),
            fillcolor='rgba(40, 167, 69, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Risk Factor Radar (0=Optimal, 100=High Risk)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("ℹ️ No lab values available for visualization. Add measurements to see risk analysis.")
    
    st.markdown("---")
    
    # Timeline/trend placeholder
    st.subheader("📅 Analysis History")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Timeline chart
        fig = px.line(
            history_df, 
            x='timestamp', 
            y='confidence',
            title='Prediction Confidence Over Time',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Confidence",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No analysis history yet. Previous analyses will appear here.")

def display_full_report(report):
    """Display complete JSON report"""
    st.header("📄 Complete Health Report")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{report['patient_info']['age']} years")
    
    with col2:
        st.metric("Gender", report['patient_info']['gender'])
    
    with col3:
        if report['patient_info'].get('bmi'):
            st.metric("BMI", f"{report['patient_info']['bmi']} kg/m²")
    
    with col4:
        st.metric("Analysis Date", report['metadata']['timestamp'].split('T')[0])
    
    st.markdown("---")
    
    # JSON display
    st.subheader("📋 Detailed Report Data")
    st.json(report)
    
    st.markdown("---")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=report_json,
            file_name=f"health_report_{report['prediction']['disease'].replace(' ', '_')}_{report['metadata']['timestamp'][:10]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text summary download
        summary = generate_text_summary(report)
        st.download_button(
            label="📥 Download Summary (TXT)",
            data=summary,
            file_name=f"health_summary_{report['metadata']['timestamp'][:10]}.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_text_summary(report):
    """Generate plain text summary of report"""
    
    summary = "="*70 + "\n"
    summary += "PATIENT HEALTH ANALYSIS REPORT\n"
    summary += "="*70 + "\n\n"
    
    summary += f"Generated: {report['metadata']['timestamp']}\n"
    summary += f"Model: {report['metadata']['model_version']}\n\n"
    
    summary += "PATIENT INFORMATION\n"
    summary += "-"*70 + "\n"
    summary += f"Age: {report['patient_info']['age']} years\n"
    summary += f"Gender: {report['patient_info']['gender']}\n"
    if report['patient_info'].get('bmi'):
        summary += f"BMI: {report['patient_info']['bmi']} kg/m²\n"
    summary += f"Symptoms: {report['patient_info'].get('symptoms', 'N/A')}\n\n"
    
    summary += "PREDICTION RESULTS\n"
    summary += "-"*70 + "\n"
    summary += f"Predicted Condition: {report['prediction']['disease']}\n"
    summary += f"Confidence: {report['prediction']['confidence']*100:.1f}%\n\n"
    
    summary += "Top 3 Predictions:\n"
    for disease, prob in report['prediction']['top_3_predictions'].items():
        summary += f"  • {disease}: {prob*100:.1f}%\n"
    summary += "\n"
    
    summary += "POPULATION COMPARISON\n"
    summary += "-"*70 + "\n"
    summary += f"Reference Population: {report['population_analysis']['sample_size']:,} participants\n"
    summary += f"Prevalence: {report['population_analysis']['prevalence']}\n\n"
    
    if report['population_analysis']['comparisons']:
        summary += "Clinical Markers:\n"
        for marker, data in report['population_analysis']['comparisons'].items():
            summary += f"\n{marker.replace('_', ' ').title()}:\n"
            summary += f"  Your value: {data.get('user_value', data.get('user_systolic', 'N/A'))}\n"
            summary += f"  Target: {data['target']}\n"
            summary += f"  Status: {data['status']}\n"
    
    summary += "\n" + "="*70 + "\n"
    summary += "RECOMMENDATIONS\n"
    summary += "="*70 + "\n\n"
    summary += report['recommendations']
    
    return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()