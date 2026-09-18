LIMIT=.07650849781930447
LABELS=['scalar-only fit does not reach the scalar gate on the 40-image nested fit; stop as optimization/capacity inconclusive','fixed source-only early stopping does not establish image-held scalar transfer under the current 28-D EnergyHead','nested source-only checkpoint selection passes the scalar gate; early-stopping regularization is supported as a viable source-validation mechanism']
def partition(parent):
 images=sorted(parent['image_ids']['train'],key=lambda i:parent['group_indices'][str(i)]);assert len(images)==48 and len(set(images))==48
 sides={s:[i for rank,i in enumerate(images) if (rank%6==5)==(s=='selector')] for s in ['fit','selector']};assert len(sides['fit'])==40 and len(sides['selector'])==8 and not set(sides['fit'])&set(sides['selector'])
 rows={s:[i for i in parent['row_indices']['train'] if parent['canonical'][i]['image_id'] in ids] for s,ids in sides.items()};banks={s:list(dict.fromkeys(parent['canonical'][i]['bank_index'] for i in ids)) for s,ids in rows.items()};assert sorted(rows['fit']+rows['selector'])==parent['row_indices']['train']
 return dict(sorted_parent_image_ids=images,parent_image_group_indices=[parent['group_indices'][str(i)] for i in images],image_ids=sides,row_indices=rows,bank_indices=banks,selector_sorted_ranks=list(range(5,48,6)))
def select(fit,selector):
 eligible=[i+1 for i,h in enumerate(fit) if h<=LIMIT]
 if not eligible:return dict(eligible_epochs=[],selected_epoch=None,classification=LABELS[0])
 e=min(eligible,key=lambda e:(selector[e-1],e));return dict(eligible_epochs=eligible,selected_epoch=e,fit_huber=fit[e-1],selector_huber=selector[e-1],fit_margin=LIMIT-fit[e-1],selector_margin=LIMIT-selector[e-1],classification=LABELS[1] if selector[e-1]>LIMIT else LABELS[2])
