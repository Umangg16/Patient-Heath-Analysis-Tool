"""
Complete mapping of predicted diseases to NHANES profiles
Based on actual NHANES 2021-2023 data availability
"""

COMPREHENSIVE_DISEASE_MAPPING = {
    # ========================================================================
    # DIRECT NHANES MATCHES (Have specific profiles)
    # ========================================================================
    'Diabetes': 'diabetes',  # ✅ 11.6% prevalence
    'Hypertension': 'hypertension',  # ✅ 35.0% prevalence
    'Heart Failure': 'heart_failure',  # ✅ 4.4% prevalence
    'Kidney Failure': 'kidney_disease',  # ✅ 6.9% elevated creatinine
    'Bronchial Asthma': 'copd_respiratory',  # ✅ 7.3% COPD (related)
    'Jaundice': 'liver_disease',  # ✅ 5.5% liver conditions
    
    # ========================================================================
    # CATEGORY-BASED MAPPING (Use related profiles)
    # ========================================================================
    'Peptic Ulcer Disease': 'prediabetes',  # GI/metabolic overlap
    
    # ========================================================================
    # NO NHANES DATA (Use healthy reference + LLM knowledge)
    # ========================================================================
    'Psoriasis': 'healthy_reference',  # Autoimmune/dermatologic (not tracked)
    'Pneumonia': 'copd_respiratory',  # Respiratory (acute infection, use COPD as base)
    'Dimorphic Hemorrhoids': 'healthy_reference',  # GI/vascular (not tracked)
    'Arthritis': 'healthy_reference',  # Musculoskeletal (MCQ160A available but not processed)
    'Urinary Tract Infection': 'healthy_reference',  # Genitourinary (acute, not tracked)
}

# Clinical context for diseases without direct NHANES profiles
DISEASE_CLINICAL_CONTEXT = {
    'Psoriasis': {
        'category': 'Autoimmune/Dermatologic',
        'key_concerns': ['Immune system', 'Inflammation', 'Skin barrier', 'Cardiovascular risk'],
        'lifestyle_focus': ['Anti-inflammatory diet', 'Stress management', 'Vitamin D', 'Omega-3'],
        'note': 'Associated with increased CVD risk - monitor metabolic health'
    },
    
    'Pneumonia': {
        'category': 'Respiratory Infection',
        'key_concerns': ['Immune recovery', 'Lung function', 'Hydration', 'Nutrition'],
        'lifestyle_focus': ['Rest and recovery', 'Adequate protein', 'Hydration', 'Gradual return to activity'],
        'note': 'Acute infection - focus on recovery and preventing recurrence'
    },
    
    'Dimorphic Hemorrhoids': {
        'category': 'Gastrointestinal/Vascular',
        'key_concerns': ['Fiber intake', 'Constipation prevention', 'Vascular health'],
        'lifestyle_focus': ['High-fiber diet (25-30g/day)', 'Adequate hydration', 'Avoid straining', 'Regular exercise'],
        'note': 'Primarily lifestyle-managed condition'
    },
    
    'Arthritis': {
        'category': 'Musculoskeletal/Inflammatory',
        'key_concerns': ['Joint health', 'Inflammation', 'Mobility', 'Pain management'],
        'lifestyle_focus': ['Anti-inflammatory foods', 'Omega-3 fatty acids', 'Weight management', 'Low-impact exercise'],
        'note': 'Weight loss of 5-10% can significantly reduce joint pain'
    },
    
    'Urinary Tract Infection': {
        'category': 'Genitourinary Infection',
        'key_concerns': ['Hydration', 'Urinary pH', 'Recurrence prevention', 'Immune support'],
        'lifestyle_focus': ['Drink 8-10 glasses water daily', 'Cranberry products', 'Proper hygiene', 'Probiotic support'],
        'note': 'Acute infection - focus on treatment completion and prevention'
    },
    
    'Peptic Ulcer Disease': {
        'category': 'Gastrointestinal',
        'key_concerns': ['H. pylori', 'Stomach acid', 'NSAID use', 'Stress'],
        'lifestyle_focus': ['Avoid NSAIDs', 'Limit alcohol/caffeine', 'Regular meals', 'Stress management'],
        'note': 'Often H. pylori-related - medical treatment essential'
    }
}

def get_profile_for_disease(disease_name: str) -> tuple:
    """
    Get NHANES profile key and clinical context for a disease
    
    Returns:
        (profile_key, clinical_context or None)
    """
    profile_key = COMPREHENSIVE_DISEASE_MAPPING.get(disease_name, 'healthy_reference')
    clinical_context = DISEASE_CLINICAL_CONTEXT.get(disease_name, None)
    
    return profile_key, clinical_context