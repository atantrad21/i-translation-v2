from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
import traceback
import time
from datetime import datetime
import json
import base64

app = Flask(__name__)
CORS(app)

# Configure maximum file size (10GB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

# Analytics storage - Use /tmp directory which exists on Railway
ANALYTICS_FILE = '/tmp/analytics_data.json'

def initialize_analytics():
    """Initialize analytics data file"""
    if not os.path.exists(ANALYTICS_FILE):
        analytics_data = {
            'total_conversions': 0,
            'ct_to_mri_count': 0,
            'mri_to_ct_count': 0,
            'dicom_uploads': 0,
            'total_file_uploads': 0,
            'file_types': {},
            'conversion_times': [],
            'errors': 0,
            'sessions': [],
            'created_at': datetime.now().isoformat(),
            'owner_email': 'atantrad@gmail.com'
        }
        save_analytics(analytics_data)
    return load_analytics()

def load_analytics():
    """Load analytics data from file"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
        else:
            # File doesn't exist, create it
            return initialize_analytics()
    except Exception as e:
        print(f"Error loading analytics: {e}")
        # Return default analytics structure
        return {
            'total_conversions': 0,
            'ct_to_mri_count': 0,
            'mri_to_ct_count': 0,
            'dicom_uploads': 0,
            'total_file_uploads': 0,
            'file_types': {},
            'conversion_times': [],
            'errors': 0,
            'sessions': [],
            'created_at': datetime.now().isoformat(),
            'owner_email': 'atantrad@gmail.com'
        }

def save_analytics(data):
    """Save analytics data to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving analytics: {e}")

def track_event(event_type, event_data):
    """Track analytics event"""
    try:
        analytics = load_analytics()
        
        if event_type == 'conversion_started':
            analytics['total_file_uploads'] += 1
            
            # Track file type
            file_type = event_data.get('file_type', 'unknown')
            if file_type not in analytics['file_types']:
                analytics['file_types'][file_type] = 0
            analytics['file_types'][file_type] += 1
            
            # Track DICOM uploads
            if event_data.get('is_dicom', False):
                analytics['dicom_uploads'] += 1
        
        elif event_type == 'conversion_completed':
            analytics['total_conversions'] += 1
            
            # Track conversion type
            conversion_type = event_data.get('conversion_type', '')
            if conversion_type == 'ct_to_mri':
                analytics['ct_to_mri_count'] += 1
            elif conversion_type == 'mri_to_ct':
                analytics['mri_to_ct_count'] += 1
            
            # Track conversion time
            conversion_time = event_data.get('conversion_time_ms', 0)
            analytics['conversion_times'].append({
                'time_ms': conversion_time,
                'conversion_type': conversion_type,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 1000 conversion times
            if len(analytics['conversion_times']) > 1000:
                analytics['conversion_times'] = analytics['conversion_times'][-1000:]
        
        elif event_type == 'conversion_error':
            analytics['errors'] += 1
        
        elif event_type == 'session_start':
            analytics['sessions'].append({
                'start_time': datetime.now().isoformat(),
                'user_agent': event_data.get('user_agent', 'unknown')
            })
            
            # Keep only last 1000 sessions
            if len(analytics['sessions']) > 1000:
                analytics['sessions'] = analytics['sessions'][-1000:]
        
        # Update last activity
        analytics['last_activity'] = datetime.now().isoformat()
        
        save_analytics(analytics)
        
        # Log to console
        print(f"📊 Analytics: {event_type} - {event_data}")
        
    except Exception as e:
        print(f"Error tracking event: {e}")

def convert_dicom_to_png(dicom_bytes):
    """
    Convert DICOM file to PNG format
    This is the key conversion step before any image processing
    """
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        
        # Read DICOM from bytes
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        
        # Get pixel array
        pixel_array = ds.pixel_array
        
        # Apply VOI LUT (Value of Interest Lookup Table) for proper windowing
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except:
            pass
        
        # Normalize to 0-255 range
        pixel_array = pixel_array.astype(float)
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0)
        
        pixel_array = pixel_array.astype(np.uint8)
        
        # Convert to PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(pixel_array, mode='L')
        else:
            # RGB image
            image = Image.fromarray(pixel_array)
        
        # Extract metadata
        metadata = {
            'PatientName': str(ds.get('PatientName', 'Unknown')),
            'StudyDate': str(ds.get('StudyDate', 'Unknown')),
            'Modality': str(ds.get('Modality', 'Unknown')),
            'ImageSize': f"{ds.Rows}x{ds.Columns}" if hasattr(ds, 'Rows') else 'Unknown'
        }
        
        return image, metadata
        
    except Exception as e:
        raise Exception(f"DICOM to PNG conversion failed: {str(e)}")

# ============================================================================
# QUAD-GAN: 4 GENERATORS FOR MULTIPLE CONVERSIONS
# ============================================================================

def generator_1_ct_to_mri(image_array):
    """
    Generator 1: Standard soft contrast MRI-like transformation
    Characteristics: Softer contrast, slightly brighter
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    mri_like = np.power(normalized, 0.7)  # Softer contrast
    mri_like = mri_like * 1.2  # Slightly brighter
    mri_like = np.clip(mri_like, 0, 1)
    return (mri_like * 255).astype(np.uint8)

def generator_2_ct_to_mri(image_array):
    """
    Generator 2: Enhanced soft tissue contrast
    Characteristics: More aggressive soft tissue enhancement, medium brightness
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    mri_like = np.power(normalized, 0.6)  # More soft tissue contrast
    mri_like = mri_like * 1.1  # Medium brightness
    mri_like = np.clip(mri_like, 0, 1)
    return (mri_like * 255).astype(np.uint8)

def generator_3_ct_to_mri(image_array):
    """
    Generator 3: Balanced MRI-like transformation
    Characteristics: Balanced contrast, neutral brightness
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    mri_like = np.power(normalized, 0.75)  # Balanced contrast
    mri_like = mri_like * 1.05  # Neutral brightness
    mri_like = np.clip(mri_like, 0, 1)
    return (mri_like * 255).astype(np.uint8)

def generator_4_ct_to_mri(image_array):
    """
    Generator 4: High contrast MRI-like transformation
    Characteristics: Higher contrast, darker overall
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    mri_like = np.power(normalized, 0.8)  # Higher contrast
    mri_like = mri_like * 1.0  # Standard brightness
    mri_like = np.clip(mri_like, 0, 1)
    return (mri_like * 255).astype(np.uint8)

def generator_1_mri_to_ct(image_array):
    """
    Generator 1: Standard sharp contrast CT-like transformation
    Characteristics: Sharper contrast, slightly darker
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    ct_like = np.power(normalized, 1.3)  # Sharper contrast
    ct_like = ct_like * 0.9  # Slightly darker
    ct_like = np.clip(ct_like, 0, 1)
    return (ct_like * 255).astype(np.uint8)

def generator_2_mri_to_ct(image_array):
    """
    Generator 2: Enhanced bone structure emphasis
    Characteristics: Very sharp contrast, darker
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    ct_like = np.power(normalized, 1.4)  # Very sharp contrast
    ct_like = ct_like * 0.85  # Darker
    ct_like = np.clip(ct_like, 0, 1)
    return (ct_like * 255).astype(np.uint8)

def generator_3_mri_to_ct(image_array):
    """
    Generator 3: Balanced CT-like transformation
    Characteristics: Balanced sharp contrast, neutral brightness
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    ct_like = np.power(normalized, 1.25)  # Balanced sharp contrast
    ct_like = ct_like * 0.95  # Neutral brightness
    ct_like = np.clip(ct_like, 0, 1)
    return (ct_like * 255).astype(np.uint8)

def generator_4_mri_to_ct(image_array):
    """
    Generator 4: Moderate contrast CT-like transformation
    Characteristics: Moderate contrast, standard brightness
    """
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    ct_like = np.power(normalized, 1.2)  # Moderate contrast
    ct_like = ct_like * 1.0  # Standard brightness
    ct_like = np.clip(ct_like, 0, 1)
    return (ct_like * 255).astype(np.uint8)

def process_uploaded_file_multi(file, conversion_type):
    """
    Main processing function for QUAD-GAN multi-conversion:
    1. Check if DICOM -> convert to PNG
    2. Apply ALL 4 transformations (CT->MRI or MRI->CT)
    3. Return 4 results (one from each generator)
    """
    try:
        filename = file.filename.lower()
        is_dicom = filename.endswith(('.dcm', '.dicom'))
        metadata = {}
        
        # Step 1: Convert DICOM to PNG if needed
        if is_dicom:
            file_bytes = file.read()
            image, metadata = convert_dicom_to_png(file_bytes)
            print(f"✓ DICOM converted to PNG: {metadata.get('ImageSize', 'Unknown')}")
        else:
            # Regular image file (PNG, JPG, etc.)
            image = Image.open(file)
            print(f"✓ Image loaded: {image.size}")
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Step 2: Apply ALL 4 transformations
        results = []
        
        if conversion_type == 'ct_to_mri':
            print("✓ Applying CT → MRI transformations with all 4 generators...")
            generators = [
                generator_1_ct_to_mri,
                generator_2_ct_to_mri,
                generator_3_ct_to_mri,
                generator_4_ct_to_mri
            ]
        elif conversion_type == 'mri_to_ct':
            print("✓ Applying MRI → CT transformations with all 4 generators...")
            generators = [
                generator_1_mri_to_ct,
                generator_2_mri_to_ct,
                generator_3_mri_to_ct,
                generator_4_mri_to_ct
            ]
        else:
            raise ValueError("Invalid conversion type")
        
        # Apply each generator
        for i, generator_func in enumerate(generators, 1):
            print(f"  → Processing with Generator {i}...")
            transformed_array = generator_func(image_array)
            result_image = Image.fromarray(transformed_array, mode='L')
            results.append(result_image)
        
        print(f"✓ All 4 conversions complete: {conversion_type}")
        return results, metadata, is_dicom
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise Exception(f"Processing failed: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'I-Translation API is running',
        'version': '2.0.0 - QUAD-GAN Multi-Conversion',
        'author': 'Atantra Das Gupta',
        'credits': 'Created with the help of Scispace',
        'generators': 4
    }), 200

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
        analytics = load_analytics()
        
        # Calculate average conversion time
        avg_conversion_time = 0
        if analytics['conversion_times']:
            total_time = sum(ct['time_ms'] for ct in analytics['conversion_times'])
            avg_conversion_time = total_time / len(analytics['conversion_times'])
        
        # Calculate popular features
        popular_conversion = 'ct_to_mri' if analytics['ct_to_mri_count'] >= analytics['mri_to_ct_count'] else 'mri_to_ct'
        
        return jsonify({
            'owner_email': analytics.get('owner_email', 'atantrad@gmail.com'),
            'summary': {
                'total_conversions': analytics['total_conversions'],
                'ct_to_mri_conversions': analytics['ct_to_mri_count'],
                'mri_to_ct_conversions': analytics['mri_to_ct_count'],
                'dicom_uploads': analytics['dicom_uploads'],
                'total_file_uploads': analytics['total_file_uploads'],
                'errors': analytics['errors'],
                'total_sessions': len(analytics['sessions']),
                'average_conversion_time_ms': round(avg_conversion_time, 2),
                'popular_conversion_type': popular_conversion
            },
            'file_types': analytics['file_types'],
            'recent_conversions': analytics['conversion_times'][-10:],  # Last 10
            'created_at': analytics.get('created_at'),
            'last_activity': analytics.get('last_activity')
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_image():
    """
    Convert medical image between CT and MRI using QUAD-GAN (4 generators)
    
    Parameters:
    - image: File (DICOM, PNG, JPG, etc.)
    - type: String ('ct_to_mri' or 'mri_to_ct')
    
    Returns: JSON with 4 converted images (base64 encoded)
    """
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get conversion type
        conversion_type = request.form.get('type', 'ct_to_mri')
        
        if conversion_type not in ['ct_to_mri', 'mri_to_ct']:
            return jsonify({
                'error': 'Invalid conversion type',
                'valid_types': ['ct_to_mri', 'mri_to_ct']
            }), 400
        
        print(f"\n{'='*50}")
        print(f"Processing: {file.filename}")
        print(f"Conversion: {conversion_type}")
        print(f"QUAD-GAN Mode: 4 Generators")
        print(f"{'='*50}")
        
        # Track conversion start
        file_extension = file.filename.split('.')[-1].lower()
        is_dicom = file_extension in ['dcm', 'dicom']
        
        track_event('conversion_started', {
            'file_type': file_extension,
            'conversion_type': conversion_type,
            'is_dicom': is_dicom,
            'file_name': file.filename[:50],
            'user_agent': request.headers.get('User-Agent', 'unknown'),
            'multi_conversion': True,
            'generators': 4
        })
        
        # Process the file with all 4 generators
        converted_images, metadata, is_dicom_file = process_uploaded_file_multi(file, conversion_type)
        
        # Calculate conversion time
        conversion_time_ms = int((time.time() - start_time) * 1000)
        
        # Track successful conversion
        track_event('conversion_completed', {
            'conversion_type': conversion_type,
            'file_type': file_extension,
            'is_dicom': is_dicom_file,
            'conversion_time_ms': conversion_time_ms,
            'multi_conversion': True,
            'generators': 4
        })
        
        # Convert all 4 images to base64
        conversions = []
        for i, img in enumerate(converted_images, 1):
            img_io = io.BytesIO()
            img.save(img_io, 'PNG', optimize=True)
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            
            conversions.append({
                'generator': i,
                'image': f'data:image/png;base64,{img_base64}',
                'description': self._get_generator_description(conversion_type, i)
            })
        
        # Prepare response
        response_data = {
            'status': 'success',
            'conversion_type': conversion_type,
            'generators_used': 4,
            'conversion_time_ms': conversion_time_ms,
            'conversions': conversions,
            'metadata': metadata if metadata else {},
            'is_dicom': is_dicom_file
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        error_msg = str(e)
        conversion_time_ms = int((time.time() - start_time) * 1000)
        
        # Track error
        track_event('conversion_error', {
            'error_message': error_msg,
            'conversion_type': request.form.get('type', 'unknown'),
            'conversion_time_ms': conversion_time_ms
        })
        
        print(f"✗ Error in /convert: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

def _get_generator_description(conversion_type, generator_num):
    """Get description for each generator"""
    if conversion_type == 'ct_to_mri':
        descriptions = {
            1: "Standard soft contrast MRI-like transformation",
            2: "Enhanced soft tissue contrast",
            3: "Balanced MRI-like transformation",
            4: "High contrast MRI-like transformation"
        }
    else:  # mri_to_ct
        descriptions = {
            1: "Standard sharp contrast CT-like transformation",
            2: "Enhanced bone structure emphasis",
            3: "Balanced CT-like transformation",
            4: "Moderate contrast CT-like transformation"
        }
    return descriptions.get(generator_num, "Unknown generator")

@app.route('/track-session', methods=['POST'])
def track_session():
    """Track user session"""
    try:
        data = request.get_json() or {}
        track_event('session_start', {
            'user_agent': request.headers.get('User-Agent', 'unknown'),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        })
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    analytics = load_analytics()
    
    return jsonify({
        'name': 'I-Translation API - QUAD-GAN',
        'version': '2.0.0',
        'description': 'Medical Image Converter - CT ↔ MRI with 4 Generators',
        'author': 'Atantra Das Gupta',
        'credits': 'Created with the help of Scispace',
        'owner_email': 'atantrad@gmail.com',
        'features': [
            'QUAD-GAN: 4 simultaneous conversions per upload',
            'DICOM to PNG conversion',
            'CT to MRI transformation (4 variations)',
            'MRI to CT transformation (4 variations)',
            'Metadata extraction',
            'Analytics tracking'
        ],
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /convert': 'Convert medical images (returns 4 results)',
            'GET /analytics': 'Get usage analytics',
            'POST /track-session': 'Track user session'
        },
        'supported_formats': {
            'input': ['DICOM (.dcm, .dicom)', 'PNG', 'JPG', 'JPEG', 'BMP', 'TIFF'],
            'output': ['PNG (4 variations per conversion)']
        },
        'conversion_types': {
            'ct_to_mri': 'Convert CT scan to 4 MRI-like images (different generators)',
            'mri_to_ct': 'Convert MRI scan to 4 CT-like images (different generators)'
        },
        'generators': {
            'total': 4,
            'description': 'Each conversion uses all 4 generators to produce 4 different results'
        },
        'usage_stats': {
            'total_conversions': analytics.get('total_conversions', 0),
            'total_sessions': len(analytics.get('sessions', []))
        }
    }), 200

# Initialize analytics on startup
try:
    initialize_analytics()
    print("✓ Analytics initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize analytics: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
    {'='*60}
    🏥 I-Translation API Server - QUAD-GAN Multi-Conversion
    {'='*60}
    Author: Atantra Das Gupta
    Credits: Created with the help of Scispace
    Owner Email: atantrad@gmail.com
    
    Server running on: http://0.0.0.0:{port}
    
    Features:
    ✓ QUAD-GAN: 4 Generators for Multiple Conversions
    ✓ DICOM to PNG conversion
    ✓ CT → MRI transformation (4 variations)
    ✓ MRI → CT transformation (4 variations)
    ✓ Metadata extraction
    ✓ Analytics tracking
    
    Each conversion now produces 4 different results!
    
    Analytics Dashboard: http://0.0.0.0:{port}/analytics
    
    Ready to accept requests!
    {'='*60}
    """)
    app.run(host='0.0.0.0', port=port, debug=False)
