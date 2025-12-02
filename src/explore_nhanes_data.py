import pandas as pd
from pathlib import Path

class NHANESDiseasesExplorer:
    """Explore all diseases and conditions available in NHANES 2021-2023"""
    
    def __init__(self, data_path='data/processed/nhanes_2021_2023_integrated.csv'):
        self.data = pd.read_csv(data_path)
        
    def explore_all_disease_columns(self):
        """Find all disease-related columns"""
        print("="*70)
        print("AVAILABLE DISEASE/CONDITION DATA IN NHANES 2021-2023")
        print("="*70)
        
        # Get all columns that look like disease indicators
        disease_cols = [col for col in self.data.columns if 'has_' in col.lower()]
        
        print(f"\nFound {len(disease_cols)} disease indicator columns:\n")
        
        for col in sorted(disease_cols):
            if col in self.data.columns:
                count = self.data[col].sum()
                total = self.data[col].notna().sum()
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {col:35s}: {count:5.0f} / {total:5.0f} ({pct:5.1f}%)")
        
        return disease_cols
    
    def check_questionnaire_data(self):
        """Check what additional conditions are available in Medical Conditions questionnaire"""
        print("\n" + "="*70)
        print("EXPLORING MEDICAL CONDITIONS QUESTIONNAIRE (MCQ)")
        print("="*70)
        
        # These are from the MCQ dataset - let's see what we have
        mcq_conditions = {
            'MCQ010': 'Ever been told you have asthma',
            'MCQ035': 'Still have asthma',
            'MCQ160A': 'Doctor ever told you had arthritis',
            'MCQ160B': 'Ever told had congestive heart failure',
            'MCQ160C': 'Ever told had coronary heart disease',
            'MCQ160D': 'Ever told had angina',
            'MCQ160E': 'Ever told had heart attack',
            'MCQ160F': 'Ever told had stroke',
            'MCQ160L': 'Ever told you had liver condition',
            'MCQ160M': 'Ever told you had thyroid problem',
            'MCQ160P': 'Ever told had COPD/emphysema/chronic bronchitis',
            'MCQ220': 'Ever told you had cancer or malignancy',
        }
        
        print("\nPotential conditions from MCQ questionnaire:")
        for code, description in mcq_conditions.items():
            print(f"  {code}: {description}")
        
        print("\n💡 We can create additional profiles from raw MCQ data!")
        
    def check_kidney_data(self):
        """Check for kidney-related markers"""
        print("\n" + "="*70)
        print("KIDNEY DISEASE INDICATORS")
        print("="*70)
        
        kidney_markers = ['creatinine', 'blood_urea_nitrogen', 'uric_acid']
        
        print("\nAvailable kidney function markers:")
        for marker in kidney_markers:
            if marker in self.data.columns:
                mean_val = self.data[marker].mean()
                std_val = self.data[marker].std()
                count = self.data[marker].notna().sum()
                print(f"  {marker:25s}: {mean_val:.2f} ± {std_val:.2f} (n={count:,})")
        
        # Calculate CKD prevalence using creatinine
        if 'creatinine' in self.data.columns:
            # Simplified CKD indicator: creatinine > 1.2 mg/dL
            elevated_creat = (self.data['creatinine'] > 1.2).sum()
            total_creat = self.data['creatinine'].notna().sum()
            pct = (elevated_creat / total_creat * 100)
            print(f"\n  Elevated creatinine (>1.2 mg/dL): {elevated_creat:,} ({pct:.1f}%)")
    
    def check_respiratory_data(self):
        """Check for respiratory/asthma indicators"""
        print("\n" + "="*70)
        print("RESPIRATORY DISEASE INDICATORS")
        print("="*70)
        
        # We have COPD from medical conditions
        if 'has_copd' in self.data.columns:
            copd_count = self.data['has_copd'].sum()
            copd_total = self.data['has_copd'].notna().sum()
            copd_pct = (copd_count / copd_total * 100)
            print(f"\n  COPD: {copd_count:,} / {copd_total:,} ({copd_pct:.1f}%)")
        
        print("\n💡 Can create profiles for:")
        print("  - Asthma (from MCQ010, MCQ035)")
        print("  - COPD (already have: has_copd)")
        print("  - General respiratory conditions")
    
    def check_liver_data(self):
        """Check for liver-related markers"""
        print("\n" + "="*70)
        print("LIVER DISEASE INDICATORS")
        print("="*70)
        
        if 'has_liver_condition' in self.data.columns:
            liver_count = self.data['has_liver_condition'].sum()
            liver_total = self.data['has_liver_condition'].notna().sum()
            liver_pct = (liver_count / liver_total * 100)
            print(f"\n  Liver condition: {liver_count:,} / {liver_total:,} ({liver_pct:.1f}%)")
        
        # Check for liver markers in biochemistry
        liver_markers = ['total_protein']  # ALT, AST not in our extract
        print("\nAvailable liver-related markers:")
        for marker in liver_markers:
            if marker in self.data.columns:
                mean_val = self.data[marker].mean()
                std_val = self.data[marker].std()
                print(f"  {marker}: {mean_val:.2f} ± {std_val:.2f}")
    
    def check_arthritis_data(self):
        """Check for arthritis indicators"""
        print("\n" + "="*70)
        print("ARTHRITIS INDICATORS")
        print("="*70)
        
        # Arthritis should be in MCQ160A but we need to check if we captured it
        print("\n⚠️  Arthritis (MCQ160A) not in processed dataset")
        print("   Need to go back to raw MCQ data and add it")
    
    def generate_expansion_recommendations(self):
        """Recommend which diseases we can actually create profiles for"""
        print("\n" + "="*70)
        print("RECOMMENDATIONS: DISEASES WE CAN ADD PROFILES FOR")
        print("="*70)
        
        recommendations = {
            '✅ Already Have': [
                'Diabetes',
                'Hypertension', 
                'Heart Failure',
                'Coronary Heart Disease',
                'Heart Attack',
                'Stroke',
                'COPD',
                'Liver Condition',
                'Thyroid Problem'
            ],
            '🔄 Can Add by Re-Processing NHANES': [
                'Arthritis (MCQ160A)',
                'Asthma (MCQ010, MCQ035)',
                'Cancer/Malignancy (MCQ220)',
                'Angina (MCQ160D)'
            ],
            '⚠️ Limited Data (Use Related Profile)': [
                'Psoriasis - No direct NHANES indicator (use general inflammatory)',
                'Pneumonia - Acute condition, not tracked in NHANES (use respiratory)',
                'Hemorrhoids - Not tracked in NHANES (use healthy reference)',
                'UTI - Acute condition (use healthy reference)',
                'Peptic Ulcer - Not tracked in NHANES (use GI-related markers)'
            ],
            '🆕 Can Create with Available Markers': [
                'Kidney Disease - Use creatinine, BUN, uric acid',
                'Metabolic Syndrome - Use metabolic_risk_score',
                'Obesity - Use BMI categories'
            ]
        }
        
        for category, diseases in recommendations.items():
            print(f"\n{category}:")
            for disease in diseases:
                print(f"  • {disease}")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("""
1. Re-process NHANES to add: Arthritis, Asthma, Cancer
2. Create kidney disease profile using creatinine/BUN
3. Create respiratory profile combining COPD + related markers
4. Create liver disease profile using available markers
5. For diseases with no NHANES data (Psoriasis, Hemorrhoids, UTI):
   - Use closest clinical category profile
   - Let LLM provide disease-specific guidance
        """)
    
    def run_full_exploration(self):
        """Run complete exploration"""
        self.explore_all_disease_columns()
        self.check_questionnaire_data()
        self.check_kidney_data()
        self.check_respiratory_data()
        self.check_liver_data()
        self.check_arthritis_data()
        self.generate_expansion_recommendations()


if __name__ == "__main__":
    explorer = NHANESDiseasesExplorer()
    explorer.run_full_exploration()