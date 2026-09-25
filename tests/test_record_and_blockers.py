"""
Unit tests for Canonical EntityRecord and Blocker Data-Contract Consistency.
Tests:
- tuple/dict representation consistency
- exact normalized-name blocking
- country+name blocking
- postal blocking
- first-token blocking
- candidate output format
- end-to-end synthetic smoke test
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.record import EntityRecord, ensure_records, get_field
from src.blocking.candidate_generator import (
    blocker_exact_norm_name,
    blocker_country_norm_name,
    blocker_first_token,
    blocker_token_overlap,
    union_candidates,
    write_candidate_pairs_tsv,
)
from src.blocking.address_blocker import blocker_country_postal_initial
from src.evaluation.blocking_metrics import evaluate_blocking


class TestEntityRecordRepresentation(unittest.TestCase):
    def test_tuple_dict_consistency(self):
        """Test that EntityRecord supports tuple indexing, dict .get(), and attribute access."""
        # 1. From tuple
        tup = ("S1-100", "Acme Corporation", "123 Main St, New York, 10001", "US")
        rec_from_tup = EntityRecord.from_any(tup)

        # Attribute access
        self.assertEqual(rec_from_tup.entity_id, "S1-100")
        self.assertEqual(rec_from_tup.business_name, "Acme Corporation")
        self.assertEqual(rec_from_tup.business_address, "123 Main St, New York, 10001")
        self.assertEqual(rec_from_tup.country, "US")

        # Tuple indexing & unpacking
        self.assertTrue(isinstance(rec_from_tup, tuple))
        self.assertEqual(rec_from_tup[0], "S1-100")
        self.assertEqual(rec_from_tup[1], "Acme Corporation")
        self.assertEqual(rec_from_tup[2], "123 Main St, New York, 10001")
        self.assertEqual(rec_from_tup[3], "US")
        eid, name, addr, country = rec_from_tup
        self.assertEqual(eid, "S1-100")
        self.assertEqual(name, "Acme Corporation")

        # Dictionary-style .get() access
        self.assertEqual(rec_from_tup.get("business_name"), "Acme Corporation")
        self.assertEqual(rec_from_tup.get("country"), "US")
        self.assertEqual(rec_from_tup.get("unknown_col", "default"), "default")

        # 2. From dict
        d = {
            "entity_id": "S2-200",
            "business_name": "Beta LLC",
            "business_address": "456 Market Rd, Delhi, 110001",
            "country": "India",
        }
        rec_from_dict = EntityRecord.from_any(d)
        self.assertEqual(rec_from_dict.entity_id, "S2-200")
        self.assertEqual(rec_from_dict.business_name, "Beta LLC")
        self.assertEqual(rec_from_dict[1], "Beta LLC")
        self.assertEqual(rec_from_dict.get("business_name"), "Beta LLC")

        # 3. Idempotent from_any and ensure_records
        rec_same = EntityRecord.from_any(rec_from_dict)
        self.assertIs(rec_same, rec_from_dict)

        rec_list = ensure_records([rec_from_tup, d, tup])
        self.assertEqual(len(rec_list), 3)
        for r in rec_list:
            self.assertTrue(isinstance(r, EntityRecord))
            self.assertTrue(isinstance(r, tuple))

    def test_get_field(self):
        """Test get_field works across tuple, dict, and EntityRecord."""
        tup = ("S1-100", "Acme Corporation", "123 Main St, New York, 10001", "US")
        d = {
            "entity_id": "S1-100",
            "business_name": "Acme Corporation",
            "business_address": "123 Main St, New York, 10001",
            "country": "US",
        }
        rec = EntityRecord.from_any(tup)
        
        for row in (tup, d, rec):
            self.assertEqual(get_field(row, 'entity_id', 0), "S1-100")
            self.assertEqual(get_field(row, 'business_name', 1), "Acme Corporation")
            self.assertEqual(get_field(row, 'business_address', 2), "123 Main St, New York, 10001")
            self.assertEqual(get_field(row, 'country', 3), "US")
        
        # Test out of bounds / missing
        self.assertEqual(get_field(("S1-200", "Only Name"), 'country', 3, "default"), "default")
        self.assertEqual(get_field({"entity_id": "S1-200"}, 'country', 3, "default"), "default")


class TestBlockersWithCanonicalRecords(unittest.TestCase):
    def setUp(self):
        # Synthetic dataset with mixed tuples and EntityRecords
        self.s1_data = [
            EntityRecord("S1-1", "Starbucks Coffee", "100 Pike St, Seattle, 98101", "US"),
            EntityRecord("S1-2", "Tata Motors Limited", "Bombay House, Mumbai, 400001", "India"),
            EntityRecord("S1-3", "Pizza Express", "12 High St, London, 90210", "US"),
        ]

        self.s2_data = [
            # Exact match for S1-1 in US
            ("S2-10", "Starbucks Coffee Corp", "100 Pike St, Seattle, 98101", "US"),
            # First token match for S1-2
            ("S2-20", "Tata Sons Enterprises", "Fort, Mumbai, 400001", "India"),
            # Different country Starbucks
            ("S2-30", "Starbucks Coffee", "Connaught Place, Delhi, 110001", "India"),
        ]

        self.s3_data = [
            # Postal match for S1-1 (same zip 98101, initial 's')
            EntityRecord("S3-100", "Seattle Roastery", "100 Pike St, Seattle, 98101", "US"),
            # Informative token match for S1-2
            EntityRecord("S3-200", "Motors World", "Pune, 411001", "India"),
        ]

    def test_exact_normalized_name_blocking(self):
        """B1 should match Starbucks Coffee regardless of Corp suffix."""
        cands, name = blocker_exact_norm_name(self.s1_data, self.s2_data, self.s3_data)
        self.assertEqual(name, "B1_exact_norm_name")
        self.assertIn("S2-10", cands["S1-1"])
        self.assertIn("S2-30", cands["S1-1"])  # B1 is global name match

    def test_country_norm_name_blocking(self):
        """B2 should match Starbucks only in US for S1-1, not India."""
        cands, name = blocker_country_norm_name(self.s1_data, self.s2_data, self.s3_data)
        self.assertEqual(name, "B2_country_norm_name")
        self.assertIn("S2-10", cands["S1-1"])
        self.assertNotIn("S2-30", cands["S1-1"])  # India Starbucks excluded for US query

    def test_postal_initial_blocking(self):
        """Postal blocker should match records sharing country, postal code, and initial."""
        cands, name = blocker_country_postal_initial(self.s1_data, self.s2_data, self.s3_data)
        self.assertEqual(name, "B_address_postal_initial")
        # S1-1 (US, 98101, 's') matches S2-10 (US, 98101, 's') and S3-100 (US, 98101, 's')
        self.assertIn("S2-10", cands["S1-1"])
        self.assertIn("S3-100", cands["S1-1"])

    def test_first_token_blocking(self):
        """B3 should match records sharing the first name token 'tata'."""
        cands, name = blocker_first_token(self.s1_data, self.s2_data, self.s3_data)
        self.assertEqual(name, "B3_first_token")
        self.assertIn("S2-20", cands["S1-2"])  # 'tata' matches 'tata'

    def test_candidate_output_format(self):
        """Verify the candidate TSV output conforms to the submission specification."""
        merged, _ = union_candidates([
            ({"S1-1": {"S2-10", "S3-100"}}, "B1"),
            ({"S1-2": {"S2-20"}}, "B2"),
            ({"S1-3": set()}, "B3"),
        ], all_s1_ids=["S1-1", "S1-2", "S1-3"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "candidate_pairs.tsv")
            write_candidate_pairs_tsv(merged, ["S1-1", "S1-2", "S1-3"], out_file)

            with open(out_file, "r", encoding="utf-8") as fh:
                lines = [line.strip().split("\t") for line in fh]

            # Header check
            self.assertEqual(lines[0], ["source1_entity_id", "candidate_entity_ids"])
            # Row count check
            self.assertEqual(len(lines), 4)

            # Check contents
            row_map = {row[0]: (row[1] if len(row) > 1 else "") for row in lines[1:]}
            self.assertEqual(row_map["S1-1"], "S2-10,S3-100")
            self.assertEqual(row_map["S1-2"], "S2-20")
            self.assertEqual(row_map["S1-3"], "")  # Empty when no candidates


class TestSyntheticEndToEndSmoke(unittest.TestCase):
    def test_full_pipeline_smoke(self):
        """End-to-end smoke test on synthetic data measuring recall."""
        s1 = [
            ("S1-A", "Alpha Traders", "1 St, NY, 10001", "US"),
            ("S1-B", "Beta Tech", "2 Rd, DL, 110001", "India"),
        ]
        s2 = [
            ("S2-A1", "Alpha Traders Inc", "1 St, NY, 10001", "US"),
            ("S2-B1", "Beta Technologies Pvt Ltd", "2 Rd, DL, 110001", "India"),
        ]
        s3 = [
            ("S3-A2", "Alpha Solutions", "1 St, NY, 10001", "US"),
        ]
        ground_truth = {
            "S1-A": {"S2-A1", "S3-A2"},
            "S1-B": {"S2-B1"},
        }

        b1_cands, _ = blocker_exact_norm_name(s1, s2, s3)
        b2_cands, _ = blocker_country_norm_name(s1, s2, s3)
        b3_cands, _ = blocker_first_token(s1, s2, s3)

        merged, _ = union_candidates([
            (b1_cands, "B1"),
            (b2_cands, "B2"),
            (b3_cands, "B3"),
        ], all_s1_ids=["S1-A", "S1-B"])

        eval_res = evaluate_blocking(merged, ground_truth)
        self.assertEqual(eval_res.total_true_pairs, 3)
        self.assertGreaterEqual(eval_res.candidate_pair_recall, 0.66)

    def test_evaluate_blocking_defensive_tuple_unwrapping(self):
        """Test evaluate_blocking safely handles a 2-tuple (candidate_map, blocker_name)."""
        s1 = [("S1-A", "Alpha Traders", "1 St, NY, 10001", "US")]
        s2 = [("S2-A1", "Alpha Traders Inc", "1 St, NY, 10001", "US")]
        s3 = []
        gt = {"S1-A": {"S2-A1"}}
        blocker_out = blocker_exact_norm_name(s1, s2, s3)  # returns (cmap, 'B1_exact_norm_name')
        res = evaluate_blocking(blocker_out, gt)
        self.assertEqual(res.candidate_pair_recall, 1.0)
        self.assertEqual(res.total_true_pairs, 1)


if __name__ == "__main__":
    unittest.main()
