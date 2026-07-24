"""
Main Data Processing Script for Advanced Recommendation Systems
Comprehensive pipeline for ALS and FAISS data preparation

Author: Enhanced for EduScroll v05
Date: September 2025

Usage:
    python process_data.py                    # Process with default settings
    python process_data.py --data-dir data   # Specify data directory
    python process_data.py --validate        # Run validation after processing
    python process_data.py --algorithms als faiss  # Process only specific algorithms
"""

import os
import sys
import argparse
import time
from typing import List, Optional

# Import our processors
from src.advanced_data_processor import AdvancedProcessData, process_data_for_algorithms
from src.als_processor import ALSDataProcessor
from src.faiss_processor import FAISSDataProcessor


def process_data_comprehensive(data_dir: str = "data", 
                             algorithms: Optional[List[str]] = None,
                             validate: bool = False) -> None:
    """
    Comprehensive data processing pipeline for all recommendation algorithms.
    
    Args:
        data_dir: Base directory containing raw data
        algorithms: List of algorithms to process for ('als', 'faiss', 'lightfm', 'all')
        validate: Whether to run validation after processing
    """
    
    if algorithms is None:
        algorithms = ['all']
        
    print("=" * 80)
    print("EDUSCROLL ADVANCED DATA PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Processing algorithms: {', '.join(algorithms)}")
    print(f"Validation enabled: {validate}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Initialize base processor
        print("\n[STEP 1] Initializing base data processor...")
        
        base_processor = AdvancedProcessData(
            interactions_path=os.path.join(data_dir, "raw", "user_interactions.csv"),
            user_features_path=os.path.join(data_dir, "raw", "user_features.csv"),
            video_features_path=os.path.join(data_dir, "raw", "video_features.csv")
        )
        
        print(f"✓ Loaded {len(base_processor.df)} interactions")
        print(f"✓ Loaded {len(base_processor.user_features)} user profiles")
        print(f"✓ Loaded {len(base_processor.video_features)} video profiles")
        
        # Create output directory
        processed_dir = os.path.join(data_dir, "processed_train")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Step 2: Process for each algorithm
        if 'all' in algorithms or 'als' in algorithms:
            print("\\n[STEP 2a] Processing data for ALS (Alternating Least Squares)...")
            
            als_processor = ALSDataProcessor(base_processor)
            als_data = als_processor.save_als_data(processed_dir)
            
            print(f"✓ ALS data saved with {als_data['confidence_matrix'].shape} confidence matrix")
            print(f"✓ Sparsity: {als_data['statistics']['sparsity_stats']['sparsity']:.4f}")
            
        if 'all' in algorithms or 'faiss' in algorithms:
            print("\\n[STEP 2b] Processing data for FAISS (Vector Similarity Search)...")
            
            faiss_processor = FAISSDataProcessor(base_processor)
            faiss_data = faiss_processor.save_faiss_data(processed_dir, embedding_dim=128)
            
            print(f"✓ FAISS data saved with {faiss_data['user_vectors'].shape} user vectors")
            print(f"✓ Video vectors: {faiss_data['video_vectors'].shape}")
            print(f"✓ Found {faiss_data['hashtag_info']['n_hashtags']} unique hashtags")
            
        # Step 3: Save base processed data (for backward compatibility)
        print("\\n[STEP 3] Saving base processed data...")
        base_processor.save_processed_data(processed_dir)
        
        # Step 4: Generate processing summary
        print("\\n[STEP 4] Generating processing summary...")
        
        summary = {
            'processing_time': time.time() - start_time,
            'data_statistics': {
                'total_interactions': len(base_processor.df),
                'unique_users': len(base_processor.user_mapping),
                'unique_videos': len(base_processor.video_mapping),
                'sparsity': 1 - len(base_processor.df) / (len(base_processor.user_mapping) * len(base_processor.video_mapping)),
                'avg_interactions_per_user': len(base_processor.df) / len(base_processor.user_mapping),
                'avg_interactions_per_video': len(base_processor.df) / len(base_processor.video_mapping)
            },
            'algorithms_processed': algorithms,
            'output_directory': processed_dir
        }
        
        # Save summary
        import json
        summary_path = os.path.join(processed_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"✓ Processing summary saved to {summary_path}")
        
        # Display summary
        print("\\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Processing time: {summary['processing_time']:.2f} seconds")
        print(f"Total interactions: {summary['data_statistics']['total_interactions']:,}")
        print(f"Unique users: {summary['data_statistics']['unique_users']:,}")
        print(f"Unique videos: {summary['data_statistics']['unique_videos']:,}")
        print(f"Data sparsity: {summary['data_statistics']['sparsity']:.4f}")
        print(f"Avg interactions per user: {summary['data_statistics']['avg_interactions_per_user']:.1f}")
        print(f"Avg interactions per video: {summary['data_statistics']['avg_interactions_per_video']:.1f}")
        print(f"Output directory: {summary['output_directory']}")
        
        # Step 5: Validation (if requested)
        if validate:
            print("\\n[STEP 5] Running validation...")
            
            from Recommend_v05.tests.test_data_processing import DataProcessingValidator
            
            validator = DataProcessingValidator(data_dir)
            validation_results = validator.run_full_validation()
            
            # Save validation report
            validation_path = os.path.join(processed_dir, "validation_report.json")
            validator.save_validation_report(validation_path)
            
            if validation_results['errors']:
                print("⚠️  Validation completed with errors. Check validation report for details.")
            else:
                print("✅ All validations passed successfully!")
                
        print("\\n" + "=" * 80)
        print("🎉 DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Instructions for next steps
        print("\\nNEXT STEPS:")
        
        if 'all' in algorithms or 'als' in algorithms:
            print("\\n📊 ALS (Collaborative Filtering):")
            print("   - Use processed_train/als_enhanced/ for implicit.als.AlternatingLeastSquares")
            print("   - Load confidence_matrix.npz for training")
            print("   - Use train/test split for evaluation")
            
        if 'all' in algorithms or 'faiss' in algorithms:
            print("\\n🔍 FAISS (Vector Search):")
            print("   - Use processed_train/faiss_enhanced/ for similarity search")
            print("   - Load user_vectors.npy and video_vectors.npy")
            print("   - Vectors are normalized for cosine similarity")
            
        print("\\n📁 File Structure:")
        print(f"   {processed_dir}/")
        print("   ├── als_enhanced/           # ALS matrices and mappings")
        print("   ├── faiss_enhanced/         # FAISS vectors and components") 
        print("   ├── original/               # Backward-compatible formats")
        print("   ├── processing_summary.json # Processing statistics")
        print("   └── validation_report.json  # Validation results (if run)")
        
    except Exception as e:
        print(f"\\n❌ ERROR: Data processing failed!")
        print(f"Error details: {str(e)}")
        
        import traceback
        print("\\nFull traceback:")
        traceback.print_exc()
        
        sys.exit(1)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Advanced data processing for EduScroll recommendation systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_data.py                           # Process all algorithms
  python process_data.py --validate               # Process all + validation
  python process_data.py --algorithms als faiss   # Process only ALS and FAISS
  python process_data.py --data-dir ../data       # Custom data directory
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Base directory containing raw data (default: data)"
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="*",
        choices=["all", "als", "faiss"],
        default=["all"],
        help="Algorithms to process data for (default: all)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after processing"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Minimize output (only errors and final summary)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' not found!")
        print("Make sure the directory exists and contains 'raw' subfolder with data files.")
        sys.exit(1)
        
    # Check for required raw data files
    required_files = [
        "user_interactions.csv",
        "user_features.csv", 
        "video_features.csv"
    ]
    
    raw_dir = os.path.join(args.data_dir, "raw")
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        print("❌ Error: Required data files not found:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\\nMake sure all required CSV files are in the raw data directory.")
        sys.exit(1)
        
    # Run processing
    try:
        process_data_comprehensive(
            data_dir=args.data_dir,
            algorithms=args.algorithms,
            validate=args.validate
        )
    except KeyboardInterrupt:
        print("\\n⚠️  Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()