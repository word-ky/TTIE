import copy
import unittest

from ttie.natural import VIEW_NAMES, CONDITIONS
from ttie.prototype_audit import apply_decisions, attach_regions, localization, summarize_prototypes


def perfect_rows():
    rows=[]
    for image_id in (1,2):
        for condition in CONDITIONS:
            for view in VIEW_NAMES:
                kind=None
                if condition=='homogeneous_dark':kind='dark'
                elif condition=='homogeneous_bright':kind='bright'
                elif condition in ('left_right','quadrants') and view!='full':
                    dark_views=('top_left','bottom_left') if condition=='left_right' else ('top_left','bottom_right')
                    kind='dark' if view in dark_views else 'bright'
                d=float(kind=='dark');b=float(kind=='bright')
                rows.append(dict(image_id=image_id,split='evaluation',condition=condition,view=view,
                                 d_dark=d,d_bright=b,zero_d_dark=d,zero_d_bright=b))
    apply_decisions(rows,dict(tau=[.5,.5]),dict(tau=[.5,.5]))
    return rows


class PrototypeAuditTests(unittest.TestCase):
    def test_mixed_labels_and_all_view_denominators(self):
        rows=perfect_rows();attach_regions(rows)
        mixed=[r for r in rows if r['region_true_type'] is not None]
        self.assertEqual(len(mixed),16)
        self.assertTrue(all(r['view']!='full' for r in mixed))
        result=localization(mixed)
        self.assertEqual(result['dark']['view_count'],8)
        self.assertEqual(result['bright']['correct_activation_recall'],1)
        dark=[r for r in mixed if r['region_true_type']=='dark']
        dark[0]['type']='bright';dark[1]['active']=False
        changed=localization(mixed)['dark']
        self.assertEqual(changed['correct_activation_recall'],6/8)
        self.assertEqual(changed['wrong_type_activation_rate'],1/8)
        self.assertEqual(changed['correct_type_among_active'],6/7)

    def test_metadata_changes_only_offline_localization(self):
        a=perfect_rows();b=copy.deepcopy(a)
        for row in b:
            if row['condition']=='quadrants':row['condition']='left_right'
            row['image_id']=999
        keys=('d_dark','d_bright','type','active','zero_d_dark','zero_d_bright','zero_type','zero_active')
        apply_decisions(b,dict(tau=[.5,.5]),dict(tau=[.5,.5]))
        attach_regions(a);attach_regions(b)
        self.assertEqual([[r[k] for k in keys] for r in a],[[r[k] for k in keys] for r in b])
        self.assertNotEqual([r['region_true_type'] for r in a],[r['region_true_type'] for r in b])

    def test_gate_requires_mixed_recall_and_low_wrong_activation(self):
        rows=perfect_rows();attach_regions(rows)
        self.assertTrue(summarize_prototypes(rows)['qualifies_later_pilot'])
        for row in rows:
            if row['region_true_type']=='bright':row['type']='dark'
        report=summarize_prototypes(rows)
        self.assertFalse(report['qualifies_later_pilot'])
        self.assertEqual(set(report['failed_criteria']),{'mixed_bright_recall','mixed_bright_wrong'})


if __name__=='__main__':unittest.main()
