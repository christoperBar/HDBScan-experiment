"""
INTELLIGENT FACE CLUSTERING WITH ADVANCED CACHING

This script extracts face embeddings from images and clusters them using DBSCAN.
It features an intelligent caching system that:

1. AUTOMATICALLY SAVES extracted face data to avoid re-processing
2. VALIDATES cache integrity and detects when source images change  
3. LOADS cached data instantly on subsequent runs
4. SHOWS detailed cache status and information

CONFIGURATION:
- Set FORCE_REEXTRACT=True to ignore cache and re-extract all faces
- Modify SOURCE_FOLDER to change the input image directory
- Modify CACHE_DIR to change where cache files are stored

CACHE FILES:
- faces.npy: Face embedding vectors
- metadata.pkl: Face metadata (paths, locations, etc.)  
- cache_info.pkl: Cache validation and creation info

USAGE:
- First run: Extracts faces and caches them (slow)
- Subsequent runs: Loads cached faces instantly (fast)
- Focus on clustering methodology without re-extraction delays!
"""

from deepface import DeepFace

import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from PIL import Image
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
import time
import hashlib
import shutil


def show_cache_status(cache_dir="coba_cache"):
    """
    Display detailed information about the current cache status
    """
    print("🔍 CACHE STATUS REPORT")
    print("=" * 40)
    
    faces_path = os.path.join(cache_dir, "faces.npy")
    metadata_path = os.path.join(cache_dir, "metadata.pkl")
    cache_info_path = os.path.join(cache_dir, "cache_info.pkl")
    
    if not os.path.exists(cache_dir):
        print("❌ Cache directory does not exist")
        return
    
    # Check core cache files
    has_faces = os.path.exists(faces_path)
    has_metadata = os.path.exists(metadata_path)
    has_cache_info = os.path.exists(cache_info_path)
    
    print(f"📁 Cache Directory: {cache_dir}")
    print(f"📄 Faces file: {'✅ Exists' if has_faces else '❌ Missing'}")
    print(f"📄 Metadata file: {'✅ Exists' if has_metadata else '❌ Missing'}")
    print(f"📄 Cache info: {'✅ Exists' if has_cache_info else '❌ Missing'}")
    
    if has_faces and has_metadata:
        try:
            # Load and show cache contents
            faces = np.load(faces_path)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            print(f"\n📊 CACHED DATA:")
            print(f"   • Face embeddings: {len(faces)}")
            
            # Only show dimensions if faces exist
            if len(faces) > 0:
                print(f"   • Face dimensions: {faces.shape[1]}D")
            else:
                print(f"   • Face dimensions: N/A (empty cache)")
                
            print(f"   • Metadata entries: {len(metadata)}")
            print(f"   • File sizes: faces={os.path.getsize(faces_path)/1024/1024:.1f}MB, "
                  f"metadata={os.path.getsize(metadata_path)/1024/1024:.1f}MB")
            
            # Show cache info if available
            if has_cache_info:
                try:
                    with open(cache_info_path, "rb") as f:
                        cache_info = pickle.load(f)
                    
                    print(f"\n📅 CACHE DETAILS:")
                    print(f"   • Created: {cache_info.get('created_time', 'Unknown')}")
                    print(f"   • Source files: {cache_info.get('folder_stats', {}).get('total_files', 'Unknown')}")
                    print(f"   • Extraction params: {cache_info.get('extraction_params', {})}")
                    
                except:
                    print("\n⚠️  Cache info file corrupted")
            
            if len(faces) > 0:
                print(f"\n✅ Cache is ready to use!")
            else:
                print(f"\n⚠️  Cache is empty - no faces found")
            
        except Exception as e:
            print(f"\n❌ Error reading cache: {e}")
    else:
        print(f"\n❌ Cache is incomplete - missing required files")
    
    print("\n💡 To force re-extraction, set FORCE_REEXTRACT=True in the script")


def get_folder_stats(folder_path):
    """
    Get statistics about folder structure for cache validation
    """
    stats = {
        'total_files': 0,
        'total_size': 0,
        'folder_structure': {},
        'last_modified': 0
    }
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    try:
                        file_stat = os.stat(file_path)
                        stats['total_files'] += 1
                        stats['total_size'] += file_stat.st_size
                        stats['last_modified'] = max(stats['last_modified'], file_stat.st_mtime)
                        
                        # Track folder structure
                        rel_path = os.path.relpath(root, folder_path)
                        if rel_path not in stats['folder_structure']:
                            stats['folder_structure'][rel_path] = 0
                        stats['folder_structure'][rel_path] += 1
                    except:
                        continue
    except:
        pass
    
    return stats


def extract_faces_from_event_folder(event_folder_path, cache_dir="coba_cache", cropped_dir="cropted_faces", min_face_size=27, force_reextract=False):
    """
    Extract faces from event folder with robust caching system
    
    Args:
        event_folder_path: Path to folder containing images (single dir or nested albums)
        cache_dir: Directory to store cached face embeddings  
        cropped_dir: Directory to store cropped face images
        min_face_size: Minimum face size to include
        force_reextract: If True, ignore cache and re-extract all faces
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    faces_path = os.path.join(cache_dir, "faces.npy")
    metadata_path = os.path.join(cache_dir, "metadata.pkl")
    cache_info_path = os.path.join(cache_dir, "cache_info.pkl")

    # Check if we should use cached data
    use_cache = not force_reextract and os.path.exists(faces_path) and os.path.exists(metadata_path)
    
    if use_cache:
        try:
            # Validate cache integrity
            print("🔍 Checking cached face data...")
            
            # Load cache info for validation
            cache_info = {}
            if os.path.exists(cache_info_path):
                try:
                    with open(cache_info_path, "rb") as f:
                        cache_info = pickle.load(f)
                except:
                    print("⚠️  Cache info corrupted, will re-extract")
                    use_cache = False
            
            if use_cache:
                # Check if source folder structure changed
                current_stats = get_folder_stats(event_folder_path)
                cached_stats = cache_info.get('folder_stats', {})
                
                if current_stats != cached_stats:
                    print("⚠️  Source images changed, cache invalid - will re-extract")
                    use_cache = False
                else:
                    # Load cached data
                    print("✅ Loading cached face data...")
                    faces = np.load(faces_path)
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                    
                    print(f"📂 Loaded {len(faces)} cached face embeddings")
                    print(f"📂 Loaded {len(metadata)} cached metadata entries")
                    print(f"📂 Cache created: {cache_info.get('created_time', 'Unknown')}")
                    
                    return faces, metadata
                    
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            print("🔄 Will re-extract faces...")
            use_cache = False

    faces = []
    metadata = []

    event_id = os.path.basename(event_folder_path)

    print(f"Processing event folder: {event_folder_path}")
    
    # Check if this is a single directory or nested structure
    if not os.path.exists(event_folder_path):
        print(f"❌ Error: Source folder '{event_folder_path}' does not exist!")
        return np.array([]), []
    
    items_in_folder = os.listdir(event_folder_path)
    print(f"📂 Found {len(items_in_folder)} items in folder")
    
    # Debug: show what's in the folder
    image_files = [f for f in items_in_folder if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    subdirs = [f for f in items_in_folder if os.path.isdir(os.path.join(event_folder_path, f))]
    
    print(f"   📸 Image files: {len(image_files)}")
    print(f"   📁 Subdirectories: {len(subdirs)}")
    
    if len(image_files) > 0:
        print(f"   📝 Sample images: {image_files[:3]}{'...' if len(image_files) > 3 else ''}")
    if len(subdirs) > 0:
        print(f"   📝 Sample subdirs: {subdirs[:3]}{'...' if len(subdirs) > 3 else ''}")
    
    has_subdirs = len(subdirs) > 0
    has_images = len(image_files) > 0
    
    # Use single directory approach if we have images directly in the folder
    if has_images:
        # Single directory with images directly inside
        print("📁 Using single directory processing mode")
        total_photos = len(image_files)
        
        with tqdm(total=total_photos, desc=f"Processing Images") as pbar:
            for img_name in image_files:
                img_path = os.path.join(event_folder_path, img_name)
                try:
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet512",
                        detector_backend="retinaface",
                        align=True,
                        enforce_detection=False,
                    )
                    for face_idx, face_data in enumerate(result):
                        face_vector = face_data['embedding']
                        facial_area = face_data['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                        # Get original image dimensions
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        
                        # Check if face size matches full image size
                        if w >= 800 or h >= 800:
                            print(f"⚠️ Face size matches full image in {img_path} face {face_idx}, skipping")
                            continue
                        
                        # Also keep min size check
                        if w < min_face_size or h < min_face_size:
                            print(f"⚠️ Face too small in {img_path} face {face_idx}, skipping") 
                            continue
                        
                        padding_factor = 0.3
                        padding_x = int(w * padding_factor)
                        padding_y = int(h * padding_factor)
                        
                        # Calculate new coordinates with padding
                        new_x = max(0, x - padding_x)
                        new_y = max(0, y - padding_y)
                        new_right = min(img_width, x + w + padding_x)
                        new_bottom = min(img_height, y + h + padding_y)
                        new_w = new_right - new_x
                        new_h = new_bottom - new_y
                        # Crop face
                        img = Image.open(img_path).convert("RGB")
                        cropped_img = img.crop((new_x, new_y, new_x + new_w, new_y + new_h))

                        # Save cropped face with unique name
                        cropped_name = f"{os.path.splitext(img_name)[0]}_f-{face_idx}.jpg"
                        cropped_path = os.path.join(cropped_dir, cropped_name)
                        cropped_img.save(cropped_path)

                        faces.append(face_vector)
                        metadata.append({
                            "foto_id": cropped_path,  # Link ke cropped face
                            "album": {
                                "id": "main",
                                "name": "main",
                                "event": {
                                    "id": event_id,
                                    "name": event_id
                                }
                            },
                            "embedding": face_vector,
                            "cluster_id": None,
                            "path": img_path,          # Link ke original image
                            "facial_area": facial_area
                        })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                pbar.update(1)
    elif has_subdirs:
        # Nested directory structure (original logic)
        print("📁 Using nested album processing mode")
        total_photos = sum(len(os.listdir(os.path.join(event_folder_path, album))) 
                           for album in os.listdir(event_folder_path) 
                           if os.path.isdir(os.path.join(event_folder_path, album)))

        with tqdm(total=total_photos, desc=f"Processing Event {event_id}") as pbar:
            for album_name in os.listdir(event_folder_path):
                album_path = os.path.join(event_folder_path, album_name)
                if os.path.isdir(album_path):
                    album_id = album_name
                    for img_name in os.listdir(album_path):
                        img_path = os.path.join(album_path, img_name)
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            try:
                                result = DeepFace.represent(
                                    img_path=img_path,
                                    model_name="Facenet512",
                                    detector_backend="retinaface",
                                    align=True,
                                    enforce_detection=False,
                                )
                                for face_idx, face_data in enumerate(result):
                                    face_vector = face_data['embedding']
                                    facial_area = face_data['facial_area']
                                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                                    # Get original image dimensions
                                    img = Image.open(img_path)
                                    img_width, img_height = img.size
                                    
                                    # Check if face size matches full image size
                                    if w >= 800 or h >= 800:
                                        print(f"⚠️ Face size matches full image in {img_path} face {face_idx}, skipping")
                                        continue
                                    
                                    # Also keep min size check
                                    if w < min_face_size or h < min_face_size:
                                        print(f"⚠️ Face too small in {img_path} face {face_idx}, skipping") 
                                        continue
                                    
                                    padding_factor = 0.3
                                    padding_x = int(w * padding_factor)
                                    padding_y = int(h * padding_factor)
                                    
                                    # Calculate new coordinates with padding
                                    new_x = max(0, x - padding_x)
                                    new_y = max(0, y - padding_y)
                                    new_right = min(img_width, x + w + padding_x)
                                    new_bottom = min(img_height, y + h + padding_y)
                                    new_w = new_right - new_x
                                    new_h = new_bottom - new_y
                                    # Crop face
                                    img = Image.open(img_path).convert("RGB")
                                    cropped_img = img.crop((new_x, new_y, new_x + new_w, new_y + new_h))

                                    # Save cropped face with unique name
                                    cropped_name = f"{os.path.splitext(img_name)[0]}_f-{face_idx}.jpg"
                                    cropped_path = os.path.join(cropped_dir, cropped_name)
                                    cropped_img.save(cropped_path)

                                    faces.append(face_vector)
                                    metadata.append({
                                        "foto_id": cropped_path,  # Link ke cropped face
                                        "album": {
                                            "id": album_id,
                                            "name": album_name,
                                            "event": {
                                                "id": event_id,
                                                "name": event_id
                                            }
                                        },
                                        "embedding": face_vector,
                                        "cluster_id": None,
                                        "path": img_path,          # Link ke original image
                                        "facial_area": facial_area
                                    })
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")
                        pbar.update(1)
    else:
        print("❌ No images or subdirectories found in the source folder!")
        print(f"   Please check if '{event_folder_path}' contains image files")
        print(f"   Supported formats: .jpg, .jpeg, .png, .bmp")
        return np.array([]), []

    # Save extracted faces to cache with validation info
    print(f"\n💾 Saving face data to cache...")
    try:
        faces_array = np.array(faces)
        
        # Save faces and metadata
        np.save(faces_path, faces_array)
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        # Save cache validation info
        cache_info = {
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_faces': len(faces),
            'num_metadata': len(metadata),
            'folder_stats': get_folder_stats(event_folder_path),
            'extraction_params': {
                'min_face_size': min_face_size,
                'model_name': "Facenet512",
                'detector_backend': "retinaface"
            }
        }
        
        with open(cache_info_path, "wb") as f:
            pickle.dump(cache_info, f)
        
        print(f"✅ Cached {len(faces)} face embeddings")
        print(f"✅ Cached {len(metadata)} metadata entries") 
        print(f"✅ Cache validation info saved")
        print(f"📁 Cache location: {cache_dir}")
        
    except Exception as e:
        print(f"⚠️  Error saving cache: {e}")
        print("Face extraction completed but cache may be incomplete")

    return faces_array, metadata




# ========================================
# FACE EXTRACTION WITH INTELLIGENT CACHING
# ========================================

# Configuration options
FORCE_REEXTRACT = False  # Set to True to ignore cache and re-extract all faces
SOURCE_FOLDER = "imageids/benzcak"  # Folder containing images to process
CACHE_DIR = "coba_cache"  # Cache directory

print("🚀 FACE CLUSTERING WITH INTELLIGENT CACHING")
print("=" * 50)

# Show current cache status
show_cache_status(CACHE_DIR)

# Check if you want to force re-extraction (useful for testing different parameters)
if FORCE_REEXTRACT:
    print("⚠️  FORCE_REEXTRACT=True: Will ignore cache and re-extract all faces")

print(f"\n🔄 FACE EXTRACTION/LOADING:")
print("-" * 30)

# Extract or load faces using the improved caching system
faces, foto_data = extract_faces_from_event_folder(
    event_folder_path=SOURCE_FOLDER,
    cache_dir=CACHE_DIR,
    force_reextract=FORCE_REEXTRACT
)

print(f"\n📊 FACE DATA SUMMARY:")
print(f"   • Total faces: {len(faces)}")
if len(faces) > 0:
    print(f"   • Face dimensions: {faces.shape[1]}D")
print(f"   • Metadata entries: {len(foto_data)}")

# Check if we have faces to cluster
if len(faces) == 0:
    print("❌ No faces found to cluster! Check your source folder and try re-extracting.")
    print("💡 Set FORCE_REEXTRACT=True to re-process images")
    exit()

print(f"   • Ready for clustering! 🎯")

# Perform HDBSCAN clustering
print(f"\n🎯 CLUSTERING PHASE:")
print("-" * 30)

def cluster_faces_hdbscan(faces, metadata, min_cluster_size=2, metric='euclidean', output_dir="clustered_faces"):
    """
    Cluster faces using HDBSCAN with cosine metric and organize them into folders
    
    Args:
        faces: Face embedding vectors
        metadata: Face metadata
        min_cluster_size: Minimum cluster size for HDBSCAN
        metric: Distance metric to use
        output_dir: Directory to save clustered faces
    
    Returns:
        cluster_labels: Array of cluster labels for each face
        n_clusters: Number of clusters found
        n_outliers: Number of outlier faces
    """
    # Validate input
    if len(faces) == 0:
        print("❌ Cannot cluster: No face embeddings provided")
        return np.array([]), 0, 0
    
    if len(faces) < min_cluster_size:
        print(f"⚠️  Warning: Only {len(faces)} faces available, but min_cluster_size={min_cluster_size}")
        print("   All faces will be treated as outliers")
        
        # Create outliers folder and copy all faces there
        organize_clustered_faces(metadata, [-1] * len(faces), output_dir)
        return np.array([-1] * len(faces)), 0, len(faces)
    
    print(f"🎯 Starting UMAP reduction...")
    n_neighbors=10 # Menyeimbangkan detail lokal vs. struktur global
    n_components=35 # Dimensi target (misal: 50), jauh lebih rendah dari 512
    min_dist=0.0 # Memadatkan cluster agar lebih rapat
    umap_metric="cosine" # **PENTING** untuk vector embedding!
    random_state=42
    
    print(f"n_neighbors: {n_neighbors}")
    print(f"n_components: {n_components}")
    print(f"min_dist: {min_dist}")
    print(f"metric: {umap_metric}")
    print(f"random_state: {random_state}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, # Menyeimbangkan detail lokal vs. struktur global
        n_components=n_components, # Dimensi target (misal: 45), jauh lebih rendah dari 512
        min_dist=min_dist, # Memadatkan cluster agar lebih rapat
        metric=umap_metric, # **PENTING** untuk vector embedding!
        random_state=random_state
    )
    reduced_embeddings = reducer.fit_transform(faces)
    print(f"Bentuk data embedding setelah reduksi: {reduced_embeddings.shape}")

    min_samples = 2

    print(f"\n🎯 Starting HDBSCAN clustering...")
    print(f"   • Min cluster size: {min_cluster_size}")
    print(f"   • Metric: {metric}")
    print(f"   • min_samples: {min_samples}")
    print(f"   • Total faces: {len(faces)}")
    # Initialize HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric, 
        cluster_selection_method='eom',
        min_samples=min_samples
        # cluster_selection_epsilon=0.5
    )
    
    # Perform clustering
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Calculate statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = list(cluster_labels).count(-1)
    
    print(f"✅ Clustering completed!")
    print(f"   • Found {n_clusters} clusters")
    print(f"   • {n_outliers} outlier faces")
    
    # Update metadata with cluster information
    for i, label in enumerate(cluster_labels):
        metadata[i]['cluster_id'] = int(label)
    
    # Organize faces into folders
    organize_clustered_faces(metadata, cluster_labels, output_dir)
    
    return cluster_labels, n_clusters, n_outliers


def organize_clustered_faces(metadata, cluster_labels, output_dir="clustered_faces"):
    """
    Copy cropped faces into organized cluster folders
    
    Args:
        metadata: Face metadata containing paths to cropped faces
        cluster_labels: Cluster labels for each face
        output_dir: Output directory for organized faces
    """
    print(f"\n📁 Organizing faces into clusters...")
    
    # Clean and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Group faces by cluster
    cluster_groups = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_groups[label].append(metadata[i])
    
    # Create folders and copy faces
    cluster_sizes = []
    
    for cluster_id, faces_in_cluster in cluster_groups.items():
        if cluster_id == -1:
            folder_name = "outliers"
        else:
            folder_name = f"cluster_{cluster_id:02d}"
            cluster_sizes.append(len(faces_in_cluster))
        
        cluster_dir = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy faces to cluster folder
        for face_data in faces_in_cluster:
            src_path = face_data['foto_id']
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(cluster_dir, filename)
                shutil.copy2(src_path, dst_path)
    
    # Calculate and show average cluster size (excluding outliers)
    if cluster_sizes:
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        print(f"   📊 Average: {avg_cluster_size:.1f} faces per cluster")
    
    print(f"✅ Faces organized in: {output_dir}")


# Perform HDBSCAN clustering
cluster_labels, n_clusters, n_outliers = cluster_faces_hdbscan(
    faces=faces,
    metadata=foto_data,
)

print(f"\n🎉 Clustering completed!")
print(f"   • Found {n_clusters} clusters")
print(f"   • {n_outliers} outlier faces")
print(f"   • Check the 'clustered_faces' folder for results")