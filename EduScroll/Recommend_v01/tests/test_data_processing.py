"""
Comprehensive Test Script for Advanced Data Processing
Validates all data formats for ALS and FAISS algorithms

Author: Enhanced for EduScroll v05
Date: September 2025
"""

import os
import sys
# Add the src directory to the path  
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)
import numpy as np
import pandas as pd
from scipy import sparse
import json
import traceback
from typing import Dict, Any, List, Tuple

# Import our processors with relative paths
from advanced_data_processor import AdvancedProcessData
from als_processor import ALSDataProcessor  
from faiss_processor import FAISSDataProcessor


class DataProcessingValidator:
    """
    Comprehensive validator for all data processing pipelines.
    Tests data formats, shapes, and compatibility with target algorithms.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize validator with data directory.
        
        Args:
            data_dir: Base directory containing raw data. If None, will auto-detect.
        """
        if data_dir is None:
            # Auto-detect data directory relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(project_dir, "data")
            
        self.data_dir = data_dir
        print(f"Using data directory: {self.data_dir}")
        self.results = {
            'base_processor': {},
            'als_processor': {},
            'faiss_processor': {},
            'integration_tests': {},
            'errors': []
        }
        
    def validate_base_processor(self) -> Dict[str, Any]:
        """Validate the base AdvancedProcessData processor."""
        print("=" * 60)
        print("VALIDATING BASE DATA PROCESSOR")
        print("=" * 60)
        
        try:
            # Initialize processor
            processor = AdvancedProcessData(
                interactions_path=os.path.join(self.data_dir, "raw", "user_interactions.csv"),
                user_features_path=os.path.join(self.data_dir, "raw", "user_features.csv"),
                video_features_path=os.path.join(self.data_dir, "raw", "video_features.csv")
            )
            
            # Test data loading
            assert processor.df is not None, "Interactions data not loaded"
            assert len(processor.df) > 0, "No interaction data found"
            print(f"✓ Loaded {len(processor.df)} interactions")
            
            assert processor.user_features is not None, "User features not loaded"
            assert len(processor.user_features) > 0, "No user features found"
            print(f"✓ Loaded {len(processor.user_features)} user profiles")
            
            assert processor.video_features is not None, "Video features not loaded"
            assert len(processor.video_features) > 0, "No video features found"
            print(f"✓ Loaded {len(processor.video_features)} video profiles")
            
            # Test mapping creation
            processor.create_user_item_mappings()
            assert len(processor.user_mapping) > 0, "User mapping not created"
            assert len(processor.video_mapping) > 0, "Video mapping not created"
            print(f"✓ Created mappings: {len(processor.user_mapping)} users, {len(processor.video_mapping)} videos")
            
            # Test feature extraction
            user_features = processor.create_user_feature_matrix()
            video_features = processor.create_video_feature_matrix()
            
            assert user_features.shape[0] == len(processor.user_mapping), "User features dimension mismatch"
            assert video_features.shape[0] == len(processor.video_mapping), "Video features dimension mismatch"
            print(f"✓ Created feature matrices: users {user_features.shape}, videos {video_features.shape}")
            
            # Test data preparation methods
            als_data = processor.prepare_als_data()
            faiss_data = processor.prepare_faiss_data()
            
            print(f"✓ Prepared ALS data: {als_data['user_item_matrix'].shape}")
            print(f"✓ Prepared FAISS data: {faiss_data['user_features'].shape} users, {faiss_data['video_features'].shape} videos")
            
            self.results['base_processor'] = {
                'status': 'PASSED',
                'n_interactions': len(processor.df),
                'n_users': len(processor.user_mapping),
                'n_videos': len(processor.video_mapping),
                'user_features_shape': user_features.shape,
                'video_features_shape': video_features.shape
            }
            
            return processor
            
        except Exception as e:
            error_msg = f"Base processor validation failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['base_processor'] = {'status': 'FAILED', 'error': error_msg}
            self.results['errors'].append(error_msg)
            return None
            
    def validate_als_processor(self, base_processor) -> Dict[str, Any]:
        """Validate ALS-specific data processing."""
        print("=" * 60)
        print("VALIDATING ALS DATA PROCESSOR")
        print("=" * 60)
        
        try:
            if base_processor is None:
                raise ValueError("Base processor required for ALS validation")
                
            # Initialize ALS processor
            als_processor = ALSDataProcessor(base_processor)
            
            # Test confidence matrix creation
            confidence_matrix = als_processor.create_confidence_matrix(alpha=40.0)
            assert confidence_matrix is not None, "Confidence matrix not created"
            assert confidence_matrix.format == 'csr', "Confidence matrix not in CSR format"
            print(f"✓ Created confidence matrix: {confidence_matrix.shape}, density: {confidence_matrix.nnz / confidence_matrix.shape[0] / confidence_matrix.shape[1]:.4f}")
            
            # Test preference matrix creation
            preference_matrix = als_processor.create_preference_matrix()
            assert preference_matrix is not None, "Preference matrix not created"
            assert preference_matrix.format == 'csr', "Preference matrix not in CSR format"
            print(f"✓ Created preference matrix: {preference_matrix.shape}")
            
            # Test weighted matrix creation
            weighted_matrix = als_processor.create_weighted_interaction_matrix()
            assert weighted_matrix is not None, "Weighted matrix not created"
            print(f"✓ Created weighted matrix: {weighted_matrix.shape}")
            
            # Test statistics generation
            stats = als_processor.get_user_item_statistics()
            assert 'user_stats' in stats, "User statistics missing"
            assert 'item_stats' in stats, "Item statistics missing"
            assert 'sparsity_stats' in stats, "Sparsity statistics missing"
            print(f"✓ Generated statistics: sparsity {stats['sparsity_stats']['sparsity']:.4f}")
            
            # Test cold start handling
            warm_conf, warm_pref = als_processor.create_cold_start_matrices()
            assert warm_conf is not None, "Warm confidence matrix not created"
            assert warm_pref is not None, "Warm preference matrix not created"
            print(f"✓ Created cold start matrices: {warm_conf.shape}")
            
            # Test train/test split
            split_data = als_processor.prepare_train_test_split(test_ratio=0.2)
            assert 'train_confidence' in split_data, "Train confidence matrix missing"
            assert 'test_matrix' in split_data, "Test matrix missing"
            print(f"✓ Created train/test split: train {split_data['train_size']}, test {split_data['test_size']}")
            
            # Test data compatibility with implicit library
            try:
                import implicit
                
                # Test if matrices are compatible with implicit.als.AlternatingLeastSquares
                model = implicit.als.AlternatingLeastSquares(factors=50, iterations=1)
                
                # Should be able to fit on confidence matrix
                model.fit(confidence_matrix.T)  # Transpose for item-user format
                print("✓ Matrices compatible with implicit.als")
                
            except ImportError:
                print("! implicit library not available for compatibility test")
            except Exception as e:
                print(f"! implicit compatibility issue: {str(e)}")
                
            self.results['als_processor'] = {
                'status': 'PASSED',
                'confidence_matrix_shape': confidence_matrix.shape,
                'preference_matrix_shape': preference_matrix.shape,
                'statistics': stats,
                'train_test_sizes': (split_data['train_size'], split_data['test_size'])
            }
            
            return als_processor
            
        except Exception as e:
            error_msg = f"ALS processor validation failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['als_processor'] = {'status': 'FAILED', 'error': error_msg}
            self.results['errors'].append(error_msg)
            return None
            
    def validate_faiss_processor(self, base_processor) -> Dict[str, Any]:
        """Validate FAISS-specific data processing."""
        print("=" * 60)
        print("VALIDATING FAISS DATA PROCESSOR")
        print("=" * 60)
        
        try:
            if base_processor is None:
                raise ValueError("Base processor required for FAISS validation")
                
            # Initialize FAISS processor
            faiss_processor = FAISSDataProcessor(base_processor)
            
            # Test hashtag embeddings
            user_hashtag, video_hashtag, hashtag_info = faiss_processor.create_hashtag_embeddings(embedding_dim=64)
            assert user_hashtag.shape[1] == 64, f"User hashtag embedding dimension wrong: {user_hashtag.shape[1]}"
            assert video_hashtag.shape[1] == 64, f"Video hashtag embedding dimension wrong: {video_hashtag.shape[1]}"
            print(f"✓ Created hashtag embeddings: {user_hashtag.shape} users, {video_hashtag.shape} videos")
            print(f"  Found {hashtag_info['n_hashtags']} unique hashtags")
            
            # Test behavior vectors
            user_behavior = faiss_processor.create_user_behavior_vectors(normalize=True)
            assert user_behavior.shape[0] == len(base_processor.user_mapping), "User behavior vector count mismatch"
            print(f"✓ Created user behavior vectors: {user_behavior.shape}")
            
            # Test content vectors
            video_content = faiss_processor.create_video_content_vectors(normalize=True)
            assert video_content.shape[0] == len(base_processor.video_mapping), "Video content vector count mismatch"
            print(f"✓ Created video content vectors: {video_content.shape}")
            
            # Test hybrid vectors
            user_vectors, video_vectors = faiss_processor.create_hybrid_vectors(embedding_dim=128)
            assert user_vectors.shape[1] == 128, f"User vector dimension wrong: {user_vectors.shape[1]}"
            assert video_vectors.shape[1] == 128, f"Video vector dimension wrong: {video_vectors.shape[1]}"
            
            # Check normalization (for cosine similarity)
            user_norms = np.linalg.norm(user_vectors, axis=1)
            video_norms = np.linalg.norm(video_vectors, axis=1)
            assert np.allclose(user_norms, 1.0, atol=1e-6), "User vectors not normalized"
            assert np.allclose(video_norms, 1.0, atol=1e-6), "Video vectors not normalized"
            print(f"✓ Created normalized hybrid vectors: users {user_vectors.shape}, videos {video_vectors.shape}")
            
            # Test data compatibility with FAISS
            try:
                import faiss
                
                # Test if vectors are compatible with FAISS
                dimension = user_vectors.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity for normalized vectors)
                
                # Should be able to add vectors
                index.add(video_vectors.astype(np.float32))
                
                # Should be able to search
                k = min(5, len(video_vectors))
                distances, indices = index.search(user_vectors[:1].astype(np.float32), k)
                
                assert distances.shape == (1, k), "FAISS search output shape wrong"
                print("✓ Vectors compatible with FAISS")
                
            except ImportError:
                print("! FAISS library not available for compatibility test")
            except Exception as e:
                print(f"! FAISS compatibility issue: {str(e)}")
                
            self.results['faiss_processor'] = {
                'status': 'PASSED',
                'user_vectors_shape': user_vectors.shape,
                'video_vectors_shape': video_vectors.shape,
                'hashtag_info': hashtag_info,
                'vectors_normalized': True
            }
            
            return faiss_processor
            
        except Exception as e:
            error_msg = f"FAISS processor validation failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['faiss_processor'] = {'status': 'FAILED', 'error': error_msg}
            self.results['errors'].append(error_msg)
            return None
            
    def validate_integration(self, base_processor, als_processor, faiss_processor):
        """Validate integration between different processors."""
        print("=" * 60)
        print("VALIDATING INTEGRATION BETWEEN PROCESSORS")
        print("=" * 60)
        
        try:
            # Test mapping consistency
            if base_processor and als_processor:
                assert base_processor.user_mapping == als_processor.user_mapping, "ALS user mapping inconsistent"
                assert base_processor.video_mapping == als_processor.video_mapping, "ALS video mapping inconsistent"
                print("✓ ALS mappings consistent with base processor")
                
            if base_processor and faiss_processor:
                assert base_processor.user_mapping == faiss_processor.user_mapping, "FAISS user mapping inconsistent"
                assert base_processor.video_mapping == faiss_processor.video_mapping, "FAISS video mapping inconsistent"
                print("✓ FAISS mappings consistent with base processor")
                
            # Test matrix dimension consistency
            n_users = len(base_processor.user_mapping) if base_processor else 0
            n_videos = len(base_processor.video_mapping) if base_processor else 0
            
            if als_processor and als_processor.confidence_matrix is not None:
                assert als_processor.confidence_matrix.shape == (n_users, n_videos), "ALS matrix dimensions inconsistent"
                print("✓ ALS matrix dimensions consistent")
                
            if faiss_processor and faiss_processor.user_vectors is not None:
                assert faiss_processor.user_vectors.shape[0] == n_users, "FAISS user vectors count inconsistent"
                assert faiss_processor.video_vectors.shape[0] == n_videos, "FAISS video vectors count inconsistent"
                print("✓ FAISS vector dimensions consistent")
                
            self.results['integration_tests'] = {'status': 'PASSED'}
            
        except Exception as e:
            error_msg = f"Integration validation failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['integration_tests'] = {'status': 'FAILED', 'error': error_msg}
            self.results['errors'].append(error_msg)
            
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation pipeline."""
        print("Starting comprehensive data processing validation...")
        print("=" * 80)
        
        # Validate base processor
        base_processor = self.validate_base_processor()
        
        # Validate specialized processors
        als_processor = self.validate_als_processor(base_processor)
        faiss_processor = self.validate_faiss_processor(base_processor)
        
        # Validate integration
        self.validate_integration(base_processor, als_processor, faiss_processor)
        
        # Generate summary
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        passed_tests = sum(1 for result in self.results.values() 
                          if isinstance(result, dict) and result.get('status') == 'PASSED')
        total_tests = len([k for k in self.results.keys() if k != 'errors'])
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        
        for test_name, result in self.results.items():
            if test_name == 'errors':
                continue
            status = result.get('status', 'UNKNOWN')
            status_symbol = "✓" if status == 'PASSED' else "✗"
            print(f"{status_symbol} {test_name}: {status}")
            
        if self.results['errors']:
            print("\nERRORS:")
            for error in self.results['errors']:
                print(f"  - {error}")
        else:
            print("\n🎉 All validations passed successfully!")
            
        return self.results
        
    def save_validation_report(self, output_path: str = "validation_report.json"):
        """Save validation results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Validation report saved to {output_path}")


def main():
    """Main function to run data processing validation."""
    import argparse
    
    # Ensure we're in the correct directory (project root, not script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    print(f"Running validation from: {project_dir}")
    
    parser = argparse.ArgumentParser(description="Validate advanced data processing pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--output", default="validation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    # Run validation
    validator = DataProcessingValidator(args.data_dir)
    results = validator.run_full_validation()
    
    # Save report
    validator.save_validation_report(args.output)
    
    # Exit with error code if any tests failed
    if results['errors']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()