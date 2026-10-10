"""Pure arithmetic over final native metrics, never teacher-state accuracy."""
def summarize(rows):
    if len(rows)!=24 or any(r['status']!='complete' or r['role']!='TRAIN' for r in rows):
        raise ValueError('Summary requires the complete fixed24 TRAIN pass')
    groups={'all24':rows,**{goal:[r for r in rows if r['goal_id']==goal] for goal in ('surface','deep')}}
    result={}
    for name,group in groups.items():
        result[name]={'n':len(group),'goal_contacts':sum(r['metrics']['goal_contacted_and_retained'] for r in group),
            'saved_search_goal_contacts':sum(r['saved_search_metrics']['goal_contacted_and_retained'] for r in group),
            'mean_return':sum(r['metrics']['total_reward'] for r in group)/len(group),
            'saved_search_mean_return':sum(r['saved_search_metrics']['total_reward'] for r in group)/len(group),
            'STOP_only':sum(r['actions']==['STOP'] for r in group),
            'same_action_sequence_as_saved_search':sum(r['actions']==r['saved_search_actions'] for r in group),
            'lower_return_than_saved_search':sum(r['reward_difference_from_saved_search']<0 for r in group),
            'same_return_as_saved_search':sum(r['reward_difference_from_saved_search']==0 for r in group),
            'higher_return_than_saved_search':sum(r['reward_difference_from_saved_search']>0 for r in group),
            'mean_removed_volume_mm3':sum(r['metrics']['removed_volume_mm3'] for r in group)/len(group)}
    return result
