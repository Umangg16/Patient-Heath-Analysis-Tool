import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Import disease mapping
from disease_mapping import COMPREHENSIVE_DISEASE_MAPPING, DISEASE_CLINICAL_CONTEXT, get_profile_for_disease

class ComprehensiveRecommendationSystem:
    """
    Complete end-to-end recommendation system integrating:
    - MIMIC-IV trained Random Forest for disease prediction
    - NHANES 2021-2023 for population comparison
    - Llama LLM for personalized recommendations
    """
    
    def __init__(self, 
                 model_dir='models',
                 profiles_path='data/profiles/comprehensive_disease_profiles.json',
                 use_llm=True,
                 llama_api_url='http://localhost:11434/api/generate',
                 llama_model='llama3.2'):
        
        print("="*70)
        print("INITIALIZING COMPREHENSIVE RECOMMENDATION SYSTEM")
        print("="*70)
        
        model_dir = Path(model_dir)
        
        # Load Random Forest model
        print("\n1. Loading Random Forest model...")
        with open(model_dir / 'random_forest_model.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        print(f"   ✓ Random Forest loaded (F1: 0.8776, AUROC: 0.9847)")
        
        # Load TF-IDF vectorizer
        print("\n2. Loading TF-IDF vectorizer...")
        with open(model_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)
        print(f"   ✓ TF-IDF loaded (max_features: {self.tfidf.max_features})")
        
        # Load Column Transformer
        print("\n3. Loading Column Transformer...")
        with open(model_dir / 'column_transformer.pkl', 'rb') as f:
            self.column_transformer = pickle.load(f)
        print(f"   ✓ Column Transformer loaded")
        
        # Load Label Encoder
        print("\n4. Loading Label Encoder...")
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load disease names
        with open(model_dir / 'disease_names.json', 'r') as f:
            self.disease_names = json.load(f)
        print(f"   ✓ Disease names loaded: {len(self.disease_names)} classes")
        
        # Load NHANES disease profiles
        print("\n5. Loading NHANES disease profiles...")
        with open(profiles_path, 'r') as f:
            self.profiles = json.load(f)
        print(f"   ✓ Loaded {len(self.profiles)} disease profiles")
        
        # LLM configuration
        self.use_llm = use_llm
        self.llama_api_url = llama_api_url
        self.llama_model = llama_model
        
        if self.use_llm:
            print(f"\n6. LLM Configuration:")
            print(f"   API: {llama_api_url}")
            print(f"   Model: {llama_model}")
        else:
            print(f"\n6. LLM: Disabled (using rule-based recommendations)")
        
        print("\n" + "="*70)
        print("✅ SYSTEM INITIALIZED SUCCESSFULLY")
        print("="*70)
    
    def prepare_input_features(self, user_input: Dict) -> tuple:
        """
        Prepare input features matching your model's training format
        
        Args:
            user_input: {
                'age': int,
                'gender': str ('M' or 'F'),
                'symptoms_text': str,  # Natural language symptoms
                'glucose': float (optional),
                'hematocrit': float (optional),
                'creatinine': float (optional),
                'potassium': float (optional),
                'sodium': float (optional),
                'urea_nitrogen': float (optional)
            }
        
        Returns:
            Combined sparse feature matrix (text + structured)
        """
        # Standardize gender
        gender = user_input.get('gender', 'M').upper()
        if gender not in ['M', 'F']:
            gender = 'M'
        
        age = user_input.get('age', 40)
        
        # Create structured data DataFrame
        structured_data = pd.DataFrame({
            'gender': [gender],
            'age': [age]
        })
        
        # Add lab values (matching your training columns)
        lab_features = {}
        
        # Your model uses mean/min/max for each lab
        # We'll use the user's value for all three (mean=min=max)
        lab_mapping = {
            'glucose': 'Glucose',
            'hematocrit': 'Hematocrit',
            'creatinine': 'Creatinine',
            'potassium': 'Potassium',
            'sodium': 'Sodium',
            'urea_nitrogen': 'Urea Nitrogen'
        }
        
        for input_key, lab_name in lab_mapping.items():
            if input_key in user_input and user_input[input_key] is not None:
                value = user_input[input_key]
                structured_data[f'mean_{lab_name}'] = value
                structured_data[f'min_{lab_name}'] = value
                structured_data[f'max_{lab_name}'] = value
        
        # Text features
        symptoms_text = user_input.get('symptoms_text', '')
        if not symptoms_text and 'symptoms' in user_input:
            # Convert list to text
            symptoms_text = ' '.join(user_input['symptoms'])
        
        # Apply text cleaning (match your training pipeline)
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        import re
        
        lemmatizer = WordNetLemmatizer()
        stop = set(stopwords.words('english'))
        
        def clean_text(s: str) -> str:
            s = str(s).lower()
            s = re.sub(r'[^a-z\s]', ' ', s)
            tokens = [w for w in s.split() if w not in stop and len(w) > 2]
            lemmas = [lemmatizer.lemmatize(w) for w in tokens]
            return " ".join(lemmas)
        
        text_clean = clean_text(symptoms_text)
        
        # Transform text with TF-IDF
        text_features = self.tfidf.transform([text_clean])
        
        # Transform structured data
        structured_features = self.column_transformer.transform(structured_data)
        
        # Combine
        combined_features = hstack([text_features, structured_features])
        
        return combined_features
    
    def predict_disease(self, user_input: Dict) -> Tuple[str, float, np.ndarray, Dict]:
        """
        Step 1: Predict disease using Random Forest model
        
        Returns:
            (predicted_disease, confidence, all_probabilities, top_3_predictions)
        """
        print("\n" + "-"*70)
        print("STEP 1: DISEASE PREDICTION (Random Forest)")
        print("-"*70)
        
        # Prepare features
        features = self.prepare_input_features(user_input)
        
        # Predict
        probabilities = self.rf_model.predict_proba(features)[0]
        predicted_class_idx = self.rf_model.predict(features)[0]
        predicted_disease = self.disease_names[predicted_class_idx]
        confidence = probabilities.max()
        
        print(f"✓ Predicted Disease: {predicted_disease}")
        print(f"✓ Confidence: {confidence*100:.1f}%")
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = {}
        
        print(f"\nTop 3 Predictions:")
        for i, idx in enumerate(top_3_idx, 1):
            disease = self.disease_names[idx]
            prob = probabilities[idx]
            top_3[disease] = float(prob)
            print(f"  {i}. {disease}: {prob*100:.1f}%")
        
        return predicted_disease, confidence, probabilities, top_3
    
    def get_nhanes_profile(self, predicted_disease: str) -> Dict:
        """Step 2: Retrieve NHANES population profile"""
        print("\n" + "-"*70)
        print("STEP 2: RETRIEVING NHANES POPULATION DATA")
        print("-"*70)
        
        # Get profile mapping and clinical context
        profile_key, clinical_context = get_profile_for_disease(predicted_disease)
        
        print(f"Disease: {predicted_disease}")
        print(f"NHANES Profile: {profile_key}")
        
        # Get profile
        profile = self.profiles.get(profile_key, self.profiles.get('healthy_reference'))
        
        # Add clinical context if available
        if clinical_context:
            profile['clinical_context'] = clinical_context
            print(f"Clinical Category: {clinical_context['category']}")
            print(f"Key Concerns: {', '.join(clinical_context['key_concerns'][:3])}")
        
        print(f"✓ Profile loaded: {profile.get('sample_size', 0):,} NHANES participants")
        
        return profile
    
    def compare_user_to_population(self, user_input: Dict, profile: Dict) -> Dict:
        """Step 3: Compare user's values to NHANES population"""
        print("\n" + "-"*70)
        print("STEP 3: POPULATION COMPARISON")
        print("-"*70)
        
        comparisons = {}
        
        # Glucose comparison
        if 'glucose' in user_input and user_input['glucose'] is not None:
            glucose_data = profile.get('clinical_markers', {}).get('glucose', {})
            if glucose_data:
                user_glucose = user_input['glucose']
                pop_mean = glucose_data.get('mean', 100)
                target = glucose_data.get('target', '<100 mg/dL')
                
                status = 'normal'
                if user_glucose >= 126:
                    status = 'diabetic range'
                elif user_glucose >= 100:
                    status = 'prediabetic range'
                
                comparisons['glucose'] = {
                    'user_value': user_glucose,
                    'population_mean': round(pop_mean, 1),
                    'target': target,
                    'status': status,
                    'deviation': round(user_glucose - pop_mean, 1)
                }
                print(f"✓ Glucose: {user_glucose} mg/dL (target: {target}, status: {status})")
        
        # BMI comparison
        if 'bmi' in user_input and user_input['bmi'] is not None:
            bmi_data = profile.get('clinical_markers', {}).get('bmi', {})
            if bmi_data:
                user_bmi = user_input['bmi']
                pop_mean = bmi_data.get('mean', 27)
                
                status = 'normal'
                if user_bmi >= 30:
                    status = 'obese'
                elif user_bmi >= 25:
                    status = 'overweight'
                elif user_bmi < 18.5:
                    status = 'underweight'
                
                comparisons['bmi'] = {
                    'user_value': user_bmi,
                    'population_mean': round(pop_mean, 1),
                    'target': '18.5-24.9 kg/m²',
                    'status': status
                }
                print(f"✓ BMI: {user_bmi} kg/m² (status: {status})")
        
        # Blood pressure (if systolic_bp provided)
        if 'systolic_bp' in user_input and user_input['systolic_bp'] is not None:
            bp_data = profile.get('clinical_markers', {}).get('blood_pressure', {})
            if bp_data:
                user_systolic = user_input['systolic_bp']
                pop_mean = bp_data.get('systolic_mean', 120)
                
                status = 'normal'
                if user_systolic >= 140:
                    status = 'stage 2 hypertension'
                elif user_systolic >= 130:
                    status = 'stage 1 hypertension'
                elif user_systolic >= 120:
                    status = 'elevated'
                
                comparisons['blood_pressure'] = {
                    'user_systolic': user_systolic,
                    'user_diastolic': user_input.get('diastolic_bp', 80),
                    'population_mean': round(pop_mean, 1),
                    'target': '<120/80 mmHg',
                    'status': status
                }
                print(f"✓ BP: {user_systolic}/{user_input.get('diastolic_bp', 80)} mmHg (status: {status})")
        
        # Creatinine (kidney function)
        if 'creatinine' in user_input and user_input['creatinine'] is not None:
            creat_data = profile.get('clinical_markers', {}).get('creatinine', {})
            if creat_data:
                user_creat = user_input['creatinine']
                pop_mean = creat_data.get('mean', 0.9)
                
                status = 'normal'
                if user_creat > 1.5:
                    status = 'significantly elevated'
                elif user_creat > 1.2:
                    status = 'elevated'
                
                comparisons['creatinine'] = {
                    'user_value': user_creat,
                    'population_mean': round(pop_mean, 2),
                    'target': '<1.2 mg/dL',
                    'status': status
                }
                print(f"✓ Creatinine: {user_creat} mg/dL (status: {status})")
        
        if not comparisons:
            print("⚠️  No clinical markers provided for comparison")
        
        return comparisons
    
    def generate_llm_recommendations(self, 
                                    predicted_disease: str,
                                    confidence: float,
                                    user_input: Dict,
                                    profile: Dict,
                                    comparisons: Dict) -> str:
        """Step 4: Generate personalized recommendations"""
        print("\n" + "-"*70)
        print("STEP 4: GENERATING PERSONALIZED RECOMMENDATIONS")
        print("-"*70)
        
        if not self.use_llm:
            print("LLM disabled - using rule-based recommendations")
            return self._generate_fallback_recommendations(predicted_disease, profile, comparisons)
        
        # Build prompt
        prompt = self._build_recommendation_prompt(
            predicted_disease, confidence, user_input, profile, comparisons
        )
        
        # Call LLM
        try:
            print("Calling Llama LLM API...")
            import requests
            
            payload = {
                "model": self.llama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }
            
            response = requests.post(self.llama_api_url, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            recommendations = result['response']
            print("✓ Recommendations generated successfully")
            
        except Exception as e:
            print(f"⚠️  LLM API error: {e}")
            print("Using fallback rule-based recommendations")
            recommendations = self._generate_fallback_recommendations(predicted_disease, profile, comparisons)
        
        return recommendations
    
    def _build_recommendation_prompt(self, 
                                    predicted_disease: str,
                                    confidence: float,
                                    user_input: Dict,
                                    profile: Dict,
                                    comparisons: Dict) -> str:
        """Prompt engineering for LLM"""
        
        clinical_context = profile.get('clinical_context', {})
        
        prompt = f"""You are an expert clinical health advisor specializing in preventive medicine and personalized lifestyle interventions following AHA, ADA, CDC, and WHO guidelines.

╔══════════════════════════════════════════════════════════════════╗
║                        PATIENT PROFILE                           ║
╚══════════════════════════════════════════════════════════════════╝

PREDICTED CONDITION: {predicted_disease}
  • Model Confidence: {confidence*100:.1f}% (Random Forest, F1=87.8%, AUROC=98.5%)
  • Age: {user_input.get('age', 'Unknown')} years
  • Gender: {user_input.get('gender', 'Unknown')}
"""
        
        if 'bmi' in user_input and user_input['bmi']:
            prompt += f"  • BMI: {user_input['bmi']} kg/m²\n"
        
        # Add symptoms
        if 'symptoms_text' in user_input and user_input['symptoms_text']:
            prompt += f"\nREPORTED SYMPTOMS:\n{user_input['symptoms_text']}\n"
        elif 'symptoms' in user_input and user_input['symptoms']:
            prompt += f"\nREPORTED SYMPTOMS:\n"
            for symptom in user_input['symptoms']:
                prompt += f"  • {symptom}\n"
        
        # Clinical measurements
        prompt += f"\nCLINICAL MEASUREMENTS:\n"
        measurements = {
            'glucose': 'Glucose (mg/dL)',
            'systolic_bp': 'Systolic BP (mmHg)',
            'diastolic_bp': 'Diastolic BP (mmHg)',
            'creatinine': 'Creatinine (mg/dL)',
            'hematocrit': 'Hematocrit (%)',
            'sodium': 'Sodium (mEq/L)',
            'potassium': 'Potassium (mEq/L)',
            'urea_nitrogen': 'Blood Urea Nitrogen (mg/dL)'
        }
        
        for key, label in measurements.items():
            if key in user_input and user_input[key] is not None:
                value = user_input[key]
                prompt += f"  • {label}: {value}"
                
                # Add status from comparisons
                comp_key = key
                if key == 'systolic_bp':
                    comp_key = 'blood_pressure'
                
                if comp_key in comparisons:
                    prompt += f" ({comparisons[comp_key]['status']})"
                prompt += "\n"
        
        # Population comparison
        if comparisons:
            prompt += f"\n╔══════════════════════════════════════════════════════════════════╗\n"
            prompt += f"║         NHANES 2021-2023 POPULATION COMPARISON                   ║\n"
            prompt += f"╚══════════════════════════════════════════════════════════════════╝\n\n"
            prompt += f"Based on {profile.get('sample_size', 0):,} similar patients:\n\n"
            
            for marker, comp_data in comparisons.items():
                prompt += f"{marker.replace('_', ' ').title().upper()}:\n"
                if 'user_value' in comp_data:
                    prompt += f"  • Your value: {comp_data['user_value']}\n"
                elif 'user_systolic' in comp_data:
                    prompt += f"  • Your value: {comp_data['user_systolic']}/{comp_data['user_diastolic']} mmHg\n"
                
                prompt += f"  • Population average: {comp_data.get('population_mean', 'N/A')}\n"
                prompt += f"  • Clinical target: {comp_data['target']}\n"
                prompt += f"  • Status: {comp_data['status'].upper()}\n\n"
        
        # Disease-specific context
        if clinical_context:
            prompt += f"╔══════════════════════════════════════════════════════════════════╗\n"
            prompt += f"║           DISEASE-SPECIFIC CONSIDERATIONS                        ║\n"
            prompt += f"╚══════════════════════════════════════════════════════════════════╝\n\n"
            prompt += f"Category: {clinical_context.get('category', 'N/A')}\n"
            prompt += f"Key Concerns:\n"
            for concern in clinical_context.get('key_concerns', []):
                prompt += f"  • {concern}\n"
            prompt += f"\nPriority Focus:\n"
            for focus in clinical_context.get('lifestyle_focus', []):
                prompt += f"  • {focus}\n"
            prompt += f"\nNote: {clinical_context.get('note', '')}\n\n"
        
        # Evidence-based guidelines
        if 'recommendations' in profile:
            prompt += f"╔══════════════════════════════════════════════════════════════════╗\n"
            prompt += f"║           EVIDENCE-BASED CLINICAL GUIDELINES                     ║\n"
            prompt += f"╚══════════════════════════════════════════════════════════════════╝\n\n"
            
            for category, recs in profile['recommendations'].items():
                prompt += f"{category.replace('_', ' ').title()}:\n"
                if isinstance(recs, dict):
                    for key, value in recs.items():
                        if isinstance(value, str):
                            prompt += f"  • {key.replace('_', ' ').title()}: {value}\n"
                prompt += "\n"
        
        # Task instruction
        prompt += f"""╔══════════════════════════════════════════════════════════════════╗
║                     YOUR TASK                                    ║
╚══════════════════════════════════════════════════════════════════╝

Generate comprehensive, personalized health recommendations.

STRUCTURE YOUR RESPONSE:

## 🚨 Immediate Priorities
Top 3 most critical changes based on their specific values.

## 🥗 Dietary Recommendations
Specific foods, portions, daily limits. Be concrete.

## 🏃 Physical Activity Plan
Type, duration, frequency, progression.

## 💊 Lifestyle Modifications
Sleep, stress, habits, environmental factors.

## 📊 Monitoring & Follow-up
What to track, how often, when to see doctor.

## ⚠️ Red Flags
Symptoms requiring immediate medical attention.

TONE: Encouraging, realistic, patient-friendly. No medical jargon.

Generate recommendations now:
"""
        
        return prompt
    
    def _generate_fallback_recommendations(self, 
                                          disease: str, 
                                          profile: Dict,
                                          comparisons: Dict) -> str:
        """Rule-based recommendations when LLM unavailable"""
        
        recs = f"# Personalized Health Recommendations for {disease}\n\n"
        recs += f"*Based on NHANES 2021-2023 population data and clinical guidelines*\n\n"
        
        # Immediate priorities
        recs += "## 🚨 Immediate Priorities\n\n"
        priority_num = 0
        
        for marker, comp in comparisons.items():
            if comp.get('status') in ['elevated', 'high', 'diabetic range', 
                                      'stage 2 hypertension', 'obese', 'significantly elevated']:
                priority_num += 1
                recs += f"{priority_num}. **{marker.replace('_', ' ').title()}** is {comp['status']}\n"
                recs += f"   - Current: {comp.get('user_value', comp.get('user_systolic', 'N/A'))}\n"
                recs += f"   - Target: {comp['target']}\n\n"
        
        if priority_num == 0:
            recs += "Continue maintaining your current healthy habits.\n\n"
        
        # Add profile recommendations
        if 'recommendations' in profile:
            for category, recs_dict in profile['recommendations'].items():
                recs += f"## {category.replace('_', ' ').title()}\n\n"
                if isinstance(recs_dict, dict):
                    for key, value in recs_dict.items():
                        if isinstance(value, str):
                            recs += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                recs += "\n"
        
        recs += "\n---\n\n"
        recs += "*⚠️ Important: These are general guidelines. Please consult your healthcare provider for personalized medical advice.*\n"
        
        return recs
    
    def generate_comprehensive_report(self, user_input: Dict) -> Dict:
        """
        🎯 MAIN METHOD: Complete pipeline
        
        Example user_input:
        {
            'age': 55,
            'gender': 'M',
            'symptoms_text': 'increased thirst, frequent urination, fatigue',
            'glucose': 145,
            'systolic_bp': 142,
            'diastolic_bp': 88,
            'creatinine': 1.3,
            'hematocrit': 38.5,
            'bmi': 32.5
        }
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE HEALTH ANALYSIS & RECOMMENDATIONS")
        print("="*70)
        print(f"\nPatient: {user_input.get('gender', 'Unknown')}, Age {user_input.get('age', 'Unknown')}")
        
        # Step 1: Disease Prediction
        predicted_disease, confidence, probabilities, top_3 = self.predict_disease(user_input)
        
        # Step 2: Get NHANES Profile
        profile = self.get_nhanes_profile(predicted_disease)
        
        # Step 3: Compare to Population
        comparisons = self.compare_user_to_population(user_input, profile)
        
        # Step 4: Generate Recommendations
        recommendations = self.generate_llm_recommendations(
            predicted_disease, confidence, user_input, profile, comparisons
        )
        
        # Compile report
        report = {
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_version': 'Random Forest (F1=0.8776, AUROC=0.9847)',
                'nhanes_version': '2021-2023'
            },
            'patient_info': {
                'age': user_input.get('age'),
                'gender': user_input.get('gender'),
                'bmi': user_input.get('bmi'),
                'symptoms': user_input.get('symptoms_text', user_input.get('symptoms', []))
            },
            'prediction': {
                'disease': predicted_disease,
                'confidence': float(confidence),
                'top_3_predictions': top_3,
                'model_used': 'Random Forest'
            },
            'population_analysis': {
                'nhanes_profile': profile.get('disease', 'Unknown'),
                'sample_size': profile.get('sample_size', 0),
                'prevalence': profile.get('nhanes_prevalence', 'N/A'),
                'comparisons': comparisons
            },
            'recommendations': recommendations
        }
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        
        return report
    
    def save_report(self, report: Dict, output_dir: str = 'reports/patient_reports') -> Path:
        """Save report to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        disease = report['prediction']['disease'].replace(' ', '_')
        filename = f"health_report_{disease}_{timestamp}.json"
        
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {filepath}")
        return filepath
    
    def print_report_summary(self, report: Dict):
        """Print formatted report summary"""
        print("\n" + "="*70)
        print("PATIENT HEALTH REPORT SUMMARY")
        print("="*70)
        
        # Patient info
        print(f"\n📋 Patient Information:")
        print(f"   Age: {report['patient_info']['age']} years")
        print(f"   Gender: {report['patient_info']['gender']}")
        if report['patient_info'].get('bmi'):
            print(f"   BMI: {report['patient_info']['bmi']} kg/m²")
        
        # Prediction
        print(f"\n🔬 Analysis Results:")
        print(f"   Predicted Condition: {report['prediction']['disease']}")
        print(f"   Confidence: {report['prediction']['confidence']*100:.1f}%")
        print(f"   Model: {report['metadata']['model_version']}")
        
        print(f"\n   Top 3 Predictions:")
        for disease, prob in report['prediction']['top_3_predictions'].items():
            print(f"     • {disease}: {prob*100:.1f}%")
        
        # Population comparison
        print(f"\n📊 Population Comparison:")
        print(f"   Reference: {report['population_analysis']['sample_size']:,} NHANES participants")
        print(f"   Prevalence: {report['population_analysis']['prevalence']}")
        
        if report['population_analysis']['comparisons']:
            print(f"\n   Clinical Markers:")
            for marker, data in report['population_analysis']['comparisons'].items():
                status = data.get('status', 'unknown')
                print(f"     • {marker.replace('_', ' ').title()}: {status}")
        
        # Recommendations preview
        print(f"\n💡 Recommendations:")
        recommendations = report['recommendations']
        if isinstance(recommendations, str):
            lines = recommendations.split('\n')[:15]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            if len(recommendations.split('\n')) > 15:
                print("\n   [See full report for complete recommendations]")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_diabetes_case():
    """Example: Diabetes patient"""
    
    print("\n" + "="*70)
    print("EXAMPLE: DIABETES CASE")
    print("="*70)
    
    # Initialize system
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models',
        profiles_path='data/profiles/comprehensive_disease_profiles.json',
        use_llm=False  # Set to True if Ollama is running
    )
    
    # Patient data
    patient_data = {
        'age': 55,
        'gender': 'M',
        'symptoms_text': 'increased thirst frequent urination fatigue blurred vision',
        'glucose': 148,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'hematocrit': 38.5,
        'creatinine': 1.1,
        'sodium': 140,
        'potassium': 4.2,
        'bmi': 32.5
    }
    
    # Generate report
    report = recommender.generate_comprehensive_report(patient_data)
    
    # Print summary
    recommender.print_report_summary(report)
    
    # Save
    filepath = recommender.save_report(report)
    
    return report


def example_hypertension_case():
    """Example: Hypertension patient"""
    
    print("\n" + "="*70)
    print("EXAMPLE: HYPERTENSION CASE")
    print("="*70)
    
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models',
        profiles_path='data/profiles/comprehensive_disease_profiles.json',
        use_llm=False
    )
    
    patient_data = {
        'age': 62,
        'gender': 'F',
        'symptoms_text': 'headache chest pain dizziness difficulty concentrating',
        'systolic_bp': 152,
        'diastolic_bp': 96,
        'glucose': 105,
        'hematocrit': 42.0,
        'sodium': 142,
        'bmi': 28.5
    }
    
    report = recommender.generate_comprehensive_report(patient_data)
    recommender.print_report_summary(report)
    filepath = recommender.save_report(report)
    
    return report


def example_kidney_failure_case():
    """Example: Kidney failure patient"""
    
    print("\n" + "="*70)
    print("EXAMPLE: KIDNEY FAILURE CASE")
    print("="*70)
    
    recommender = ComprehensiveRecommendationSystem(
        model_dir='models',
        profiles_path='data/profiles/comprehensive_disease_profiles.json',
        use_llm=False
    )
    
    patient_data = {
        'age': 68,
        'gender': 'M',
        'symptoms_text': 'fatigue nausea metallic taste decreased urine output swelling legs',
        'creatinine': 2.8,
        'urea_nitrogen': 45,
        'glucose': 110,
        'systolic_bp': 155,
        'diastolic_bp': 92,
        'hematocrit': 32.0,
        'potassium': 5.2,
        'bmi': 29.0
    }
    
    report = recommender.generate_comprehensive_report(patient_data)
    recommender.print_report_summary(report)
    filepath = recommender.save_report(report)
    
    return report


if __name__ == "__main__":
    # Run all examples
    print("\n" + "="*70)
    print("RUNNING EXAMPLE CASES")
    print("="*70)
    
    print("\n\n")
    report1 = example_diabetes_case()
    
    print("\n\n")
    report2 = example_hypertension_case()
    
    print("\n\n")
    report3 = example_kidney_failure_case()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated 3 patient reports in: reports/patient_reports/")