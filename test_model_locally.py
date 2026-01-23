"""
Local Model Testing Script for Mail Guard Dashboard
This script tests your spam classifier model locally
and identifies prediction discrepancies
"""
from feature_extractor_class import TextFeatureExtractor
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

print("\n" + "=" * 70)
print("MAIL GUARD DASHBOARD - LOCAL MODEL TESTING")
print("=" * 70 + "\n")

# ============================================================================
# SECTION 0: VERIFY FILES EXIST
# ============================================================================

print("Checking if required files exist in 'models/' folder...")
required_files = [
    'models/stacking_model.pkl',
    'models/scaler.pkl',
    'models/feature_extractor.pkl',
    'models/best_threshold.pkl'
]
missing_files = []

for file in required_files:
    if os.path.exists(file):
        print(f"✓ Found: {file}")
    else:
        print(f"✗ MISSING: {file}")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ ERROR: Missing files: {', '.join(missing_files)}")
    print("\nMake sure the 'models' folder with all .pkl files is in the same directory")
    input("\nPress Enter to exit...")
    exit()

print("\n")

# ============================================================================
# SECTION 1: LOAD MODEL AND COMPONENTS
# ============================================================================

print("=" * 70)
print("LOADING MODEL AND COMPONENTS")
print("=" * 70)

try:
    # Load the trained stacking model
    with open('models/stacking_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Stacking model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    input("\nPress Enter to exit...")
    exit()

try:
    # Load the scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
    print(f"  Scaler type: {type(scaler).__name__}")
except Exception as e:
    print(f"✗ ERROR loading scaler: {e}")
    scaler = None
    print("  WARNING: Scaler not found, continuing without it")

try:
    # Load the feature extractor
    with open('models/feature_extractor.pkl', 'rb') as f:
        feature_extractor = pickle.load(f)
    print("✓ Feature extractor loaded successfully")
    print(f"  Extractor type: {type(feature_extractor).__name__}")
except Exception as e:
    print(f"✗ ERROR loading feature extractor: {e}")
    input("\nPress Enter to exit...")
    exit()

try:
    # Load the best threshold
    with open('models/best_threshold.pkl', 'rb') as f:
        best_threshold = pickle.load(f)
    print("✓ Best threshold loaded successfully")
    print(f"  Threshold value: {best_threshold}")
except Exception as e:
    print(f"✗ ERROR loading threshold: {e}")
    best_threshold = 0.5
    print(f"  WARNING: Using default threshold: {best_threshold}")

print("\n")

# ============================================================================
# SECTION 2: TEST WITH SAMPLE EMAILS
# ============================================================================

print("=" * 70)
print("TESTING WITH SAMPLE EMAILS")
print("=" * 70)

# Define test cases
test_emails = [
    {
        "name": "Test 1 - Legitimate Email",
        "text": "Hi John, How are you doing? Let's schedule a meeting next week to discuss the project. Best regards, Sarah"
    },
    {
        "name": "Test 2 - Spam Email",
        "text": "CONGRATULATIONS! You won $1,000,000!!! Click here NOW to claim your prize!!! Limited time offer!!! Act immediately!!!"
    },
    {
        "name": "Test 3 - Another Legitimate",
        "text": "The meeting has been rescheduled to Thursday at 2 PM. Please confirm your attendance."
    },
    {
        "name": "Test 4 - Suspicious Email",
        "text": "Click here to verify your account immediately or it will be suspended. Update your password now."
    }
]

results = []

for idx, test_case in enumerate(test_emails, 1):
    print(f"\n[Test {idx}] {test_case['name']}")
    print("-" * 70)
    
    email_text = test_case['text']
    
    # Step 1: Log raw input
    print(f"Raw input: {email_text[:60]}...")
    print(f"Email length: {len(email_text)} characters")
    
    try:
        # Step 2: Extract features
        features = feature_extractor.transform([email_text])
        print(f"\n✓ Feature extraction successful")
        print(f"  Features shape: {features.shape}")
        
        # Step 3: Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features)
            print(f"✓ Feature scaling successful")
        else:
            features_scaled = features
            print("⚠ Scaling skipped (scaler not loaded)")
        
        # Step 4: Make prediction
        prediction_proba = model.predict_proba(features_scaled)
        
        # Apply threshold
        spam_prob = prediction_proba
        is_spam = spam_prob >= best_threshold
        
        print(f"\n✓ Prediction successful")
        print(f"  Safe probability: {prediction_proba:.4f}")
        print(f"  Spam probability: {spam_prob:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")
        print(f"  Confidence: {max(prediction_proba, spam_prob):.4f}")
        
        # Determine label
        label = "SPAM ⚠" if is_spam else "SAFE ✓"
        
        print(f"  Final result: {label}")
        
        # Store result for summary
        results.append({
            'Test': test_case['name'],
            'Email_Preview': email_text[:50] + '...',
            'Prediction': label,
            'Safe_Prob': f"{prediction_proba:.4f}",
            'Spam_Prob': f"{spam_prob:.4f}",
            'Confidence': f"{max(prediction_proba, spam_prob):.4f}"
        })
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        print(f"  Error type: {type(e).__name__}")

print("\n")

# ============================================================================
# SECTION 3: SUMMARY TABLE
# ============================================================================

print("=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

if results:
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    # Save results to CSV for comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.csv"
    df_results.to_csv(filename, index=False)
    print(f"\n✓ Results saved to: {filename}")
    print(f"  Location: {os.path.abspath(filename)}")

print("\n")

# ============================================================================
# SECTION 4: COMPARE WITH COLAB RESULTS
# ============================================================================

print("=" * 70)
print("NEXT STEPS: COMPARE WITH COLAB")
print("=" * 70)

print(f"""
1. Copy the test emails from above to your Google Colab notebook

2. In Colab, run this code for each test email:
   -----------------------------------------------------------
   import pickle
   
   # Load all components
   model = pickle.load(open('stacking_model.pkl', 'rb'))
   scaler = pickle.load(open('scaler.pkl', 'rb'))
   feature_extractor = pickle.load(open('feature_extractor.pkl', 'rb'))
   best_threshold = pickle.load(open('best_threshold.pkl', 'rb'))
   
   # Paste one of the test emails here:
   email = "Hi John, How are you doing?..."
   
   # Process
   features = feature_extractor.transform([email])
   features_scaled = scaler.transform(features)
   proba = model.predict_proba(features_scaled)
   
   print(f"Safe: {{proba:.4f}}, Spam: {{proba:.4f}}")
   print(f"Threshold: {best_threshold}")
   print(f"Result: {{'SPAM' if proba >= best_threshold else 'SAFE'}}")
   -----------------------------------------------------------

3. Compare the results:
   ✓ Are predictions the same for each email?
   ✓ Are probabilities close (within 0.01)?
   ✓ Is the threshold being applied correctly?
   
4. If results MATCH:
   → Issue is in your Flask app or Cloud Run deployment
   → Check app.py preprocessing code
   → Check if feature extraction is done the same way
   
5. If results DIFFER:
   → Check feature extraction differences
   → Check scaler application
   → Check threshold application
""")

print("\n")

# ============================================================================
# SECTION 5: VERIFY MODEL COMPONENTS
# ============================================================================

print("=" * 70)
print("MODEL COMPONENTS INFORMATION (for debugging)")
print("=" * 70)

try:
    print(f"\nModel parameters:")
    print(f"  - Type: {type(model).__name__}")
    if hasattr(model, 'classes_'):
        print(f"  - Classes: {model.classes_}")
    if hasattr(model, 'n_features_in_'):
        print(f"  - Expected features: {model.n_features_in_}")
except Exception as e:
    print(f"Could not read model parameters: {e}")

try:
    print(f"\nFeature Extractor parameters:")
    print(f"  - Type: {type(feature_extractor).__name__}")
    if hasattr(feature_extractor, 'get_feature_names_out'):
        vocab_size = len(feature_extractor.get_feature_names_out())
        print(f"  - Vocabulary size: {vocab_size} features")
except Exception as e:
    print(f"Could not read feature extractor parameters: {e}")

try:
    print(f"\nScaler parameters:")
    print(f"  - Type: {type(scaler).__name__}")
    if hasattr(scaler, 'scale_'):
        print(f"  - Number of features scaled: {len(scaler.scale_)}")
except Exception as e:
    print(f"Could not read scaler parameters: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

# Keep window open
input("\nPress Enter to close...")
