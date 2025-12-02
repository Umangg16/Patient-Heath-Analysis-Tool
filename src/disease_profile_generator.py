import pandas as pd
import numpy as np
import json
from pathlib import Path

class EnhancedDiseaseProfileGenerator:
    """Generate profiles for ALL 13+ diseases available in NHANES 2021-2023"""
    
    def __init__(self, nhanes_data):
        self.data = nhanes_data
        print(f"Loaded NHANES data: {len(self.data):,} participants")
        
    def create_diabetes_profile(self):
        """Diabetes profile (11.6% prevalence)"""
        print("\nGenerating Diabetes profile...")
        
        diabetic = self.data[self.data['has_diabetes'] == 1].copy()
        non_diabetic = self.data[self.data['has_diabetes'] == 0].copy()
        
        profile = {
            'disease': 'Diabetes',
            'nhanes_prevalence': '11.6%',
            'sample_size': int(diabetic['has_diabetes'].sum()),
            
            'demographics': {
                'avg_age': float(diabetic['age'].mean()),
                'age_std': float(diabetic['age'].std())
            },
            
            'clinical_markers': {
                'glucose': {
                    'mean': float(diabetic['glucose'].mean()),
                    'median': float(diabetic['glucose'].median()),
                    'std': float(diabetic['glucose'].std()),
                    'target': '<100 mg/dL (fasting)',
                    'diabetes_threshold': '≥126 mg/dL'
                },
                'bmi': {
                    'mean': float(diabetic['bmi'].mean()),
                    'pct_obese': float((diabetic['bmi'] >= 30).mean() * 100),
                    'target': '18.5-24.9 kg/m²'
                },
                'blood_pressure': {
                    'systolic_mean': float(diabetic['systolic_bp'].mean()),
                    'target': '<130/80 mmHg'
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_calories': float(diabetic['calories'].mean()),
                    'avg_sugar_g': float(diabetic['sugar_g'].mean()),
                    'avg_fiber_g': float(diabetic['fiber_g'].mean()),
                    'avg_carbs_g': float(diabetic['carbs_g'].mean())
                },
                'activity': {
                    'avg_weekly_mins': float(diabetic['weekly_activity_mins'].mean()),
                    'pct_meeting_guidelines': float(diabetic['meets_activity_guidelines'].mean() * 100)
                }
            },
            
            'recommendations': {
                'glucose_control': {
                    'target': 'Fasting glucose <100 mg/dL, HbA1c <5.7%',
                    'action': 'Monitor glucose regularly, limit added sugars'
                },
                'diet': {
                    'sugar_limit': '<25g added sugars per day',
                    'fiber_goal': '>30g fiber per day',
                    'carb_strategy': '45-60g carbs per meal, focus on complex carbs'
                },
                'activity': {
                    'goal': '150 minutes moderate activity per week',
                    'strength_training': '2 days per week'
                }
            }
        }
        
        print(f"  ✓ Diabetes: {profile['sample_size']} participants")
        return profile
    
    def create_hypertension_profile(self):
        """Hypertension profile (35.0% prevalence)"""
        print("\nGenerating Hypertension profile...")
        
        htn = self.data[self.data['has_hypertension'] == 1].copy()
        
        profile = {
            'disease': 'Hypertension',
            'nhanes_prevalence': '35.0%',
            'sample_size': int(htn['has_hypertension'].sum()),
            
            'clinical_markers': {
                'blood_pressure': {
                    'systolic_mean': float(htn['systolic_bp'].mean()),
                    'diastolic_mean': float(htn['diastolic_bp'].mean()),
                    'target': '<120/80 mmHg'
                },
                'bmi': {
                    'mean': float(htn['bmi'].mean())
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_sodium_mg': float(htn['sodium_mg'].mean()),
                    'avg_potassium_mg': float(htn['potassium_mg'].mean())
                }
            },
            
            'recommendations': {
                'diet': {
                    'sodium_limit': '<2,300mg per day (ideally <1,500mg)',
                    'potassium_goal': '>3,500mg per day',
                    'dash_diet': 'Follow DASH diet pattern'
                },
                'activity': {
                    'goal': '150 minutes moderate aerobic per week',
                    'benefit': 'Can lower BP by 5-8 mmHg'
                }
            }
        }
        
        print(f"  ✓ Hypertension: {profile['sample_size']} participants")
        return profile
    
    def create_cvd_profile(self):
        """Cardiovascular disease profile (12.6% prevalence)"""
        print("\nGenerating Cardiovascular Disease profile...")
        
        cvd = self.data[self.data['has_cvd'] == 1].copy()
        
        profile = {
            'disease': 'Cardiovascular_Disease',
            'nhanes_prevalence': '12.6%',
            'sample_size': int(cvd['has_cvd'].sum()),
            'includes': 'Heart attack, stroke, heart failure, coronary disease',
            
            'clinical_markers': {
                'cholesterol': {
                    'mean': float(cvd['total_cholesterol'].mean()),
                    'target': '<200 mg/dL'
                },
                'blood_pressure': {
                    'systolic_mean': float(cvd['systolic_bp'].mean()),
                    'target': '<120/80 mmHg'
                }
            },
            
            'recommendations': {
                'diet': {
                    'sodium_limit': '<1,500mg per day',
                    'saturated_fat': '<7% of calories',
                    'fiber_goal': '>30g per day'
                },
                'lifestyle': {
                    'smoking': 'Quit immediately',
                    'alcohol': 'Limit or avoid',
                    'stress': 'Practice stress management'
                }
            }
        }
        
        print(f"  ✓ CVD: {profile['sample_size']} participants")
        return profile
    
    def create_kidney_disease_profile(self):
        """NEW: Kidney disease profile (6.9% with elevated creatinine)"""
        print("\nGenerating Kidney Disease profile...")
        
        kidney_disease = self.data[self.data['creatinine'] > 1.2].copy()
        
        profile = {
            'disease': 'Kidney_Disease',
            'nhanes_prevalence': '6.9% (elevated creatinine)',
            'sample_size': len(kidney_disease),
            
            'clinical_markers': {
                'creatinine': {
                    'mean': float(kidney_disease['creatinine'].mean()),
                    'median': float(kidney_disease['creatinine'].median()),
                    'target': '<1.2 mg/dL',
                    'concern': '>1.5 mg/dL'
                },
                'bun': {
                    'mean': float(kidney_disease['blood_urea_nitrogen'].mean()),
                    'target': '7-20 mg/dL'
                },
                'uric_acid': {
                    'mean': float(kidney_disease['uric_acid'].mean()),
                    'target': '<7.0 mg/dL (men), <6.0 mg/dL (women)'
                }
            },
            
            'recommendations': {
                'kidney_protection': {
                    'protein': 'Moderate protein (0.8-1.0g/kg body weight)',
                    'sodium_limit': '<2,000mg per day',
                    'hydration': 'Adequate water unless restricted',
                    'avoid_nsaids': 'Limit ibuprofen, naproxen'
                },
                'monitoring': {
                    'creatinine': 'Check every 3-6 months',
                    'gfr': 'Monitor kidney function (eGFR)',
                    'urine': 'Check for protein in urine'
                }
            }
        }
        
        print(f"  ✓ Kidney Disease: {profile['sample_size']} participants")
        return profile
    
    def create_copd_profile(self):
        """COPD profile (7.3% prevalence)"""
        print("\nGenerating COPD/Respiratory Disease profile...")
        
        copd = self.data[self.data['has_copd'] == 1].copy()
        
        profile = {
            'disease': 'COPD_Respiratory',
            'nhanes_prevalence': '7.3%',
            'sample_size': int(copd['has_copd'].sum()),
            
            'clinical_markers': {
                'bmi': {
                    'mean': float(copd['bmi'].mean())
                }
            },
            
            'comorbidities': {
                'pct_hypertension': float(copd['has_hypertension'].mean() * 100),
                'pct_cvd': float(copd['has_cvd'].mean() * 100)
            },
            
            'recommendations': {
                'respiratory_health': {
                    'quit_smoking': 'CRITICAL - stop smoking immediately',
                    'avoid_irritants': 'Avoid air pollution, dust, chemicals',
                    'vaccinations': 'Flu shot yearly, pneumonia vaccine'
                },
                'exercise': {
                    'pulmonary_rehab': 'Consider pulmonary rehabilitation',
                    'breathing_exercises': 'Pursed-lip breathing technique',
                    'activity': '20-30 min light activity daily'
                }
            }
        }
        
        print(f"  ✓ COPD: {profile['sample_size']} participants")
        return profile
    
    def create_liver_disease_profile(self):
        """Liver disease profile (5.5% prevalence)"""
        print("\nGenerating Liver Disease profile...")
        
        liver = self.data[self.data['has_liver_condition'] == 1].copy()
        
        profile = {
            'disease': 'Liver_Disease',
            'nhanes_prevalence': '5.5%',
            'sample_size': int(liver['has_liver_condition'].sum()),
            
            'clinical_markers': {
                'total_protein': {
                    'mean': float(liver['total_protein'].mean()),
                    'normal': '6.0-8.3 g/dL'
                },
                'bmi': {
                    'mean': float(liver['bmi'].mean())
                }
            },
            
            'recommendations': {
                'liver_protection': {
                    'alcohol': 'AVOID alcohol completely',
                    'weight_loss': 'If overweight, lose 7-10%',
                    'medications': 'Avoid acetaminophen overuse'
                },
                'diet': {
                    'limit_fat': 'Reduce saturated fats',
                    'fiber': 'High-fiber diet',
                    'coffee': 'Coffee may be beneficial (2-3 cups/day)'
                }
            }
        }
        
        print(f"  ✓ Liver Disease: {profile['sample_size']} participants")
        return profile
    
    def create_thyroid_profile(self):
        """Thyroid disease profile (13.5% prevalence)"""
        print("\nGenerating Thyroid Disease profile...")
        
        thyroid = self.data[self.data['has_thyroid_problem'] == 1].copy()
        
        profile = {
            'disease': 'Thyroid_Disease',
            'nhanes_prevalence': '13.5%',
            'sample_size': int(thyroid['has_thyroid_problem'].sum()),
            
            'recommendations': {
                'management': {
                    'medication': 'Take thyroid medication as prescribed',
                    'timing': 'Same time daily, empty stomach',
                    'monitoring': 'TSH check every 6-12 months'
                },
                'nutrition': {
                    'iodine': 'Adequate iodine (iodized salt, seafood)',
                    'selenium': 'Selenium-rich foods (Brazil nuts, fish)'
                }
            }
        }
        
        print(f"  ✓ Thyroid Disease: {profile['sample_size']} participants")
        return profile
    
    def create_heart_failure_profile(self):
        """Heart failure profile (4.4% prevalence)"""
        print("\nGenerating Heart Failure profile...")
        
        hf = self.data[self.data['has_heart_failure'] == 1].copy()
        
        profile = {
            'disease': 'Heart_Failure',
            'nhanes_prevalence': '4.4%',
            'sample_size': int(hf['has_heart_failure'].sum()),
            
            'recommendations': {
                'fluid_management': {
                    'sodium_limit': '<1,500mg per day (strict)',
                    'fluid_restriction': 'May need to limit fluids (ask doctor)',
                    'daily_weights': 'Weigh daily, report sudden gains'
                },
                'medications': {
                    'compliance': 'Take all medications as prescribed',
                    'diuretics': 'Take diuretics consistently'
                },
                'activity': {
                    'cardiac_rehab': 'Enroll in cardiac rehabilitation',
                    'gradual': 'Increase activity gradually as tolerated'
                }
            }
        }
        
        print(f"  ✓ Heart Failure: {profile['sample_size']} participants")
        return profile
    
    def create_prediabetes_profile(self):
        """Prediabetes profile (11.5% prevalence)"""
        print("\nGenerating Prediabetes profile...")
        
        prediab = self.data[self.data['has_prediabetes'] == 1].copy()
        
        profile = {
            'disease': 'Prediabetes',
            'nhanes_prevalence': '11.5%',
            'sample_size': int(prediab['has_prediabetes'].sum()),
            
            'clinical_markers': {
                'glucose': {
                    'mean': float(prediab['glucose'].mean()),
                    'prediabetes_range': '100-125 mg/dL'
                }
            },
            
            'recommendations': {
                'prevention': {
                    'weight_loss': 'Lose 5-7% body weight',
                    'activity': '150 min/week can prevent diabetes',
                    'diet': 'Reduce refined carbs, increase fiber'
                },
                'monitoring': {
                    'glucose': 'Check fasting glucose annually',
                    'hba1c': 'HbA1c test yearly'
                }
            }
        }
        
        print(f"  ✓ Prediabetes: {profile['sample_size']} participants")
        return profile
    
    def create_high_cholesterol_profile(self):
        """High cholesterol profile (36.7% prevalence)"""
        print("\nGenerating High Cholesterol profile...")
        
        high_chol = self.data[self.data['has_high_cholesterol_dx'] == 1].copy()
        
        profile = {
            'disease': 'High_Cholesterol',
            'nhanes_prevalence': '36.7%',
            'sample_size': int(high_chol['has_high_cholesterol_dx'].sum()),
            
            'clinical_markers': {
                'total_cholesterol': {
                    'mean': float(high_chol['total_cholesterol'].mean()),
                    'target': '<200 mg/dL'
                }
            },
            
            'recommendations': {
                'diet': {
                    'saturated_fat': '<7% of calories',
                    'trans_fat': 'Eliminate trans fats',
                    'fiber': '>30g per day (esp. soluble fiber)',
                    'omega3': 'Fatty fish 2x per week'
                },
                'lifestyle': {
                    'weight_loss': 'If overweight, lose 5-10%',
                    'exercise': '150 min/week aerobic activity'
                }
            }
        }
        
        print(f"  ✓ High Cholesterol: {profile['sample_size']} participants")
        return profile
    
    def create_healthy_reference_profile(self):
        """Healthy reference population"""
        print("\nGenerating Healthy Reference profile...")
        
        healthy = self.data[
            (self.data['has_diabetes'] == 0) &
            (self.data['has_hypertension'] == 0) &
            (self.data['has_cvd'] == 0) &
            (self.data['bmi'] >= 18.5) &
            (self.data['bmi'] < 30)
        ].copy()
        
        profile = {
            'disease': 'Healthy_Reference',
            'description': 'No diabetes, HTN, CVD, normal BMI',
            'sample_size': len(healthy),
            
            'clinical_markers': {
                'glucose': {
                    'mean': float(healthy['glucose'].mean()),
                    'target': '70-99 mg/dL'
                },
                'blood_pressure': {
                    'systolic_mean': float(healthy['systolic_bp'].mean()),
                    'target': '<120/80 mmHg'
                },
                'bmi': {
                    'mean': float(healthy['bmi'].mean()),
                    'target': '18.5-24.9 kg/m²'
                }
            },
            
            'lifestyle_patterns': {
                'dietary': {
                    'avg_calories': float(healthy['calories'].mean()),
                    'avg_fiber_g': float(healthy['fiber_g'].mean())
                },
                'activity': {
                    'pct_meeting_guidelines': float(healthy['meets_activity_guidelines'].mean() * 100)
                }
            }
        }
        
        print(f"  ✓ Healthy Reference: {profile['sample_size']} participants")
        return profile
    
    def generate_all_profiles(self):
        """Generate ALL 11 disease profiles from NHANES"""
        print("="*70)
        print("GENERATING COMPREHENSIVE DISEASE PROFILES FROM NHANES")
        print("="*70)
        
        profiles = {
            # Primary metabolic/cardiovascular (original 4 + expansions)
            'diabetes': self.create_diabetes_profile(),
            'prediabetes': self.create_prediabetes_profile(),
            'hypertension': self.create_hypertension_profile(),
            'cardiovascular_disease': self.create_cvd_profile(),
            'heart_failure': self.create_heart_failure_profile(),
            'high_cholesterol': self.create_high_cholesterol_profile(),
            
            # Organ-specific
            'kidney_disease': self.create_kidney_disease_profile(),
            'liver_disease': self.create_liver_disease_profile(),
            'copd_respiratory': self.create_copd_profile(),
            'thyroid_disease': self.create_thyroid_profile(),
            
            # Reference
            'healthy_reference': self.create_healthy_reference_profile()
        }
        
        # Save
        output_dir = Path('data/profiles')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'comprehensive_disease_profiles.json'
        
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        
        print(f"\n✓ Saved all profiles to: {output_path}")
        
        # Summary
        print("\n" + "="*70)
        print("PROFILE SUMMARY")
        print("="*70)
        for disease_key, profile in profiles.items():
            prevalence = profile.get('nhanes_prevalence', 'N/A')
            print(f"{profile['disease']:30s}: {profile['sample_size']:5,} participants ({prevalence})")
        
        print(f"\n✅ Total: {len(profiles)} disease profiles generated")
        
        return profiles


if __name__ == "__main__":
    # Load processed NHANES data
    print("Loading NHANES data...")
    data = pd.read_csv('data/processed/nhanes_2021_2023_integrated.csv')
    
    # Generate all profiles
    generator = EnhancedDiseaseProfileGenerator(data)
    profiles = generator.generate_all_profiles()
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE PROFILE GENERATION COMPLETE!")
    print("="*70)