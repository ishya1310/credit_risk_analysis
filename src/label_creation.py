
import pandas as pd

def map_status(status):
    if status in ['0', 'C', 'X']:
        return 0
    elif status in ['1', '2']:
        return 1
    else:
        return 2

def create_type_of_client(credit_df):
    credit_df = credit_df.copy()

    credit_df['STATUS_SEVERITY'] = credit_df['STATUS'].apply(map_status)
    credit_df['RECENT'] = credit_df['MONTHS_BALANCE'] >= -6

    agg = credit_df.groupby('ID').agg(
        mild_count=('STATUS_SEVERITY', lambda x: (x == 1).sum()),
        severe_count=('STATUS_SEVERITY', lambda x: (x == 2).sum()),
        max_severity=('STATUS_SEVERITY', 'max'),
        recent_severe=(
            'STATUS_SEVERITY',
            lambda x: ((x == 2) & credit_df.loc[x.index, 'RECENT']).any()
        )
    ).reset_index()

    def classify(row):
        if row['severe_count'] > 1 or row['recent_severe']:
            return 'Bad'
        elif row['mild_count'] > 1 or row['max_severity'] == 2:
            return 'Risky'
        else:
            return 'Trusted'

    agg['TYPE_OF_CLIENT'] = agg.apply(classify, axis=1)

    return agg[['ID', 'TYPE_OF_CLIENT']]
