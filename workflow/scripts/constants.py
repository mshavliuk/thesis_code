from pyspark.sql import functions as F


def get_features():
    # this cannot be usual const because F.col requires active spark context
    
    BLOOD_PRESSURE_FILTER = {
        'filtering': {
            'valid': F.col('value').between(0, 375),
            'action': 'remove'
        }}
    GLUCOSE_FILTER = {
        'filtering': {
            'valid': F.col('value').between(0, 2200),
            'action': 'remove'
        }}
    BILIRUBIN_FILTER = {
        'filtering': {
            'valid': F.col('value').between(0, 66),
            'action': 'remove'
        }}
    
    return (
        
        {
            'name': 'DBP',
            'codes': [8368, 220051, 225310, 8555, 8441, 220180, 8502,
                      8440, 8503, 8504, 8507, 8506, 224643, 227242],
            **BLOOD_PRESSURE_FILTER
        },
        {
            'name': 'SBP',
            'codes': [51, 220050, 225309, 6701, 455, 220179, 3313, 3315,
                      442, 3317, 3323, 3321, 224167, 227243],
            **BLOOD_PRESSURE_FILTER,
        },
        {
            'name': 'MBP',
            'codes': [52, 220052, 225312, 224, 6702, 224322, 456, 220181,
                      3312, 3314, 3316, 3322, 3320, 443],
            **BLOOD_PRESSURE_FILTER,
        },
        {
            'name': 'GCS_eye',  # FIXME: GCS are categorical data!
            'codes': [184, 220739],
            'categorical': True,
        },
        {
            'name': 'GCS_motor',
            'codes': [454, 223901],
            'categorical': True,
        },
        {
            'name': 'GCS_verbal',
            'codes': [723, 223900],
            'categorical': True,
        },
        {
            'name': 'HR',
            'codes': [211, 220045],
            'filtering': {'valid': F.col('value').between(0, 390), 'action': 'remove'},
        },
        {
            'name': 'RR',
            'codes': [618, 220210, 3603, 224689, 614, 651, 224422, 615,
                      224690, 619, 224688, 227860, 227918],
            'filtering': {'valid': F.col('value').between(0, 330), 'action': 'remove'},
        },
        {
            # Celsius
            'name': 'Temperature',
            'codes': [3655, 677, 676, 223762],
            'filtering': {'valid': F.col('value').between(14.2, 47), 'action': 'remove'},
        },
        {
            # Fahrenheit
            'name': 'Temperature',
            'codes': [223761, 678, 679, 3654],
            'select': (F.col('value') - 32) * 5 / 9,
            'filtering': {'valid': F.col('value').between(14.2, 47), 'action': 'remove'},
        },
        {
            # weight kg
            'name': 'Weight',
            'codes': [224639, 226512, 226846, 763],
            'filtering': {'valid': F.col('value').between(0, 300), 'action': 'remove'},
        },
        {
            # weight lb
            'name': 'Weight',
            'codes': [226531],
            'select': F.col('value') * 0.453592,
            'filtering': {'valid': F.col('value').between(0, 300), 'action': 'remove'},
        },
        {
            # height cm
            'name': 'Height',
            'codes': [226730],
            'filtering': {'valid': F.col('value').between(0, 275), 'action': 'remove'},
        },
        {
            'name': 'Height',
            'codes': [1394, 226707],
            'select': F.col('value') * 2.54,
            'filtering': {'valid': F.col('value').between(0, 275), 'action': 'remove'},
        },
        {
            'name': 'FiO2',
            'codes': [3420, 223835, 3422, 189, 727, 190],
            'select': F.when(F.col('value') > 1,
                             F.col('value') / 100).otherwise(F.col('value')),
            'filtering': {'valid': F.col('value').between(0.2, 1), 'action': 'remove'},
        },
        {
            'name': 'CRR',
            'codes': [3348, 115, 8377, 224308, 223951],
            'select': F.when((F.col('value') == 'Normal <3 Seconds') | (
                F.col('value') == 'Normal <3 secs'), 0)
            .when((F.col('value') == 'Abnormal >3 Seconds') | (
                F.col('value') == 'Abnormal >3 secs'), 1)
            .otherwise(None),
            'categorical': True,
        },
        {
            'name': 'Glucose (Blood)',
            'codes': [225664, 1529, 811, 807, 3745, 50809],
            **GLUCOSE_FILTER,
        },
        {
            'name': 'Glucose (Whole Blood)',
            'codes': [226537],
            **GLUCOSE_FILTER,
        },
        {
            'name': 'Glucose (Serum)',
            'codes': [220621, 50931],
            **GLUCOSE_FILTER,
        },
        {
            'name': 'Bilirubin (Total)',
            'codes': [50885],
            **BILIRUBIN_FILTER,
        },
        {
            'name': 'Bilirubin (Direct)',
            'codes': [50883],
            **BILIRUBIN_FILTER,
        },
        {
            'name': 'Bilirubin (Indirect)',
            'codes': [50884],
            **BILIRUBIN_FILTER,
        },
        {
            'name': 'O2 Saturation',
            'codes': [834, 50817, 8498, 220227, 646, 220277],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Sodium',
            'codes': [50983, 50824],
            'filtering': {'valid': F.col('value').between(0, 250), 'action': 'remove'},
        },
        {
            'name': 'Potassium',
            'codes': [50971, 50822],
            'filtering': {'valid': F.col('value').between(0, 15), 'action': 'remove'},
        },
        {
            'name': 'Magnesium',
            'codes': [50960],
            'filtering': {'valid': F.col('value').between(0, 22), 'action': 'remove'},
        },
        {
            'name': 'Phosphate',
            'codes': [50970],
            'filtering': {'valid': F.col('value').between(0, 22), 'action': 'remove'},
        },
        {
            'name': 'Calcium Total',
            'codes': [50893],
            'filtering': {'valid': F.col('value').between(0, 40), 'action': 'remove'},
        },
        {
            'name': 'Calcium Free',
            'codes': [50808],
            'filtering': {'valid': F.col('value').between(0, 10), 'action': 'remove'},
        },
        {
            'name': 'WBC',
            'codes': [51301, 51300],
            'filtering': {'valid': F.col('value').between(0, 1100), 'action': 'remove'},
        },
        {
            'name': 'Hct',
            'codes': [50810, 51221],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Hgb',
            'codes': [51222, 50811],
            'filtering': {'valid': F.col('value').between(0, 30), 'action': 'remove'},
        },
        {
            'name': 'Chloride',
            'codes': [50902, 50806],
            'filtering': {'valid': F.col('value').between(0, 200), 'action': 'remove'},
        },
        {
            'name': 'Bicarbonate',
            'codes': [50882, 50803],
            'filtering': {'valid': F.col('value').between(0, 66), 'action': 'remove'},
        },
        {
            'name': 'ALT',
            'codes': [50861],
            'filtering': {'valid': F.col('value').between(0, 11000), 'action': 'remove'},
        },
        {
            'name': 'ALP',
            'codes': [50863],
            'filtering': {'valid': F.col('value').between(0, 4000), 'action': 'remove'},
        },
        {
            'name': 'AST',
            'codes': [50878],
            'filtering': {'valid': F.col('value').between(0, 22000), 'action': 'remove'},
        },
        {
            'name': 'Albumin',
            'codes': [50862],
            'filtering': {'valid': F.col('value').between(0, 10), 'action': 'remove'},
        },
        {
            'name': 'Lactate',
            'codes': [50813],
            'filtering': {'valid': F.col('value').between(0, 33), 'action': 'remove'},
        },
        {
            'name': 'LDH',
            'codes': [50954],
            'filtering': {'valid': F.col('value').between(0, 35000), 'action': 'remove'},
        },
        {
            'name': 'SG Urine',
            'codes': [51498],
            'filtering': {'valid': F.col('value').between(0, 2), 'action': 'remove'},
        },
        {
            'name': 'pH Urine',
            'codes': [51491, 51094, 220734, 1495, 1880, 1352, 6754, 7262],
            'filtering': {'valid': F.col('value').between(0, 14), 'action': 'remove'},
        },
        {
            'name': 'pH Blood',
            'codes': [50820],
            'filtering': {'valid': F.col('value').between(0, 14), 'action': 'remove'},
        },
        {
            'name': 'PO2',
            'codes': [50821],
            'filtering': {'valid': F.col('value').between(0, 770), 'action': 'remove'},
        },
        {
            'name': 'PCO2',
            'codes': [50818],
            'filtering': {'valid': F.col('value').between(0, 220), 'action': 'remove'},
        },
        {
            'name': 'Total CO2',
            'codes': [50804],
            'filtering': {'valid': F.col('value').between(0, 65), 'action': 'remove'},
        },
        {
            'name': 'Base Excess',
            'codes': [50802],
            'filtering': {'valid': F.col('value').between(-31, 28), 'action': 'remove'},
        },
        {
            'name': 'Monocytes',
            'codes': [51254],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Basophils',
            'codes': [51146],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Eoisinophils',
            'codes': [51200],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Neutrophils',
            'codes': [51256],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Lymphocytes',
            'codes': [51244, 51245],
            'filtering': {'valid': F.col('value').between(0, 100), 'action': 'remove'},
        },
        {
            'name': 'Lymphocytes (Absolute)',
            'codes': [51133],
            'filtering': {'valid': F.col('value').between(0, 25000), 'action': 'remove'},
        },
        {
            'name': 'PT',
            'codes': [51274],
            'filtering': {'valid': F.col('value').between(0, 150), 'action': 'remove'},
        },
        {
            'name': 'PTT',
            'codes': [51275],
            'filtering': {'valid': F.col('value').between(0, 150), 'action': 'remove'},
        },
        {
            'name': 'INR',
            'codes': [51237],
            'filtering': {'valid': F.col('value').between(0, 150), 'action': 'remove'},
        },
        {
            'name': 'Anion Gap',
            'codes': [50868],
            'filtering': {'valid': F.col('value').between(0, 55), 'action': 'remove'},
        },
        {
            'name': 'BUN',
            'codes': [51006],
            'filtering': {'valid': F.col('value').between(0, 275), 'action': 'remove'},
        },
        {
            'name': 'Creatinine Blood',
            'codes': [50912],
            'filtering': {'valid': F.col('value').between(0, 66), 'action': 'remove'},
        },
        {
            'name': 'Creatinine Urine',
            'codes': [51082],
            'filtering': {'valid': F.col('value').between(0, 650), 'action': 'remove'},
        },
        {
            'name': 'MCH',
            'codes': [51248],
            'filtering': {'valid': F.col('value').between(0, 50), 'action': 'remove'},
        },
        {
            'name': 'MCHC',
            'codes': [51249],
            'filtering': {'valid': F.col('value').between(0, 50), 'action': 'remove'},
        },
        {
            'name': 'MCV',
            'codes': [51250],
            'filtering': {'valid': F.col('value').between(0, 150), 'action': 'remove'},
        },
        {
            'name': 'RDW',
            'codes': [51277],
            'filtering': {'valid': F.col('value').between(0, 37), 'action': 'remove'},
        },
        {
            'name': 'Platelet Count',
            'codes': [51265],
            'filtering': {'valid': F.col('value').between(0, 2200), 'action': 'remove'},
        },
        {
            'name': 'RBC',
            'codes': [51279],
            'filtering': {'valid': F.col('value').between(0, 14), 'action': 'remove'},
        },
        {
            'name': 'Intubated',
            'codes': [50812],
            'select': (F.when(F.col('value') == 'INTUBATED', 1)
                       .when(F.col('value') == 'NOT INTUBATED', 0)),
            'categorical': True,
        },
        
        # OUTPUTEVENTS
        
        # TODO: target column for all outputevents VALUE
        # TODO: for all outputevents, replace outliers with median
        {
            'name': 'Ultrafiltrate',
            'codes': [40286],
            'filtering': {'valid': F.col('value').between(0, 7000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Urine',
            # see get_urine_itemids
            'codes': [42817, 226560, 42666, 41510, 42119, 45991, 40061, 40288, 44911, 40405, 45927,
                      42068, 42130, 46658, 42362, 42556, 40096, 226559, 226564, 44229, 43931, 40055,
                      44325, 44253, 44834, 227701, 42001, 46732, 40065, 42592, 46748, 44676, 42507,
                      44080, 46578, 42700, 46532, 41247, 226627, 46539, 40428, 43093, 41184, 40473,
                      40056, 44824, 44837, 46755, 44132, 42676, 44278, 40534, 40715, 44103, 45415,
                      44925, 227489, 42765, 42510, 42892, 43053, 41857, 40057, 45841, 40094, 41177,
                      41868, 42463, 42366, 43897, 46727, 42111, 226631, 226561, 44506, 46804, 42042,
                      42810, 40069, 42859, 45971, 45804, 44051, 40085, 44684, 43042, 40651, 46177,
                      41922, 41839, 46180, 43987, 226565, 44313, 42209, 44752],
            'filtering': {'valid': F.col('value').between(0, 2500),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Stool',
            # see get_stool_itemids
            'codes': [40054, 40388, 40053, 40367, 40063, 40087, 40751, 40068, 40289, 40991, 41433,
                      44266, 45337, 45992, 46186, 226579, 226580, 226583],
            'filtering': {'valid': F.col('value').between(0, 4000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Chest Tube',
            # see get_chest_tube_itemids
            'codes': [41838, 45227, 42540, 43984, 41003, 42516, 42256, 42498, 45405, 42258, 42539,
                      40050, 42101, 43857, 45417, 46237, 40091, 40048, 41584, 226588, 42255, 41681,
                      45664, 226589, 40090, 45813, 45883, 40084, 40049, 42257, 41707, 40076, 226593,
                      226590, 226591, 226595, 226592],
            'filtering': {'valid': F.col('value').between(0, 2500),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Gastric',
            'codes': [40059, 40052, 226576, 226575, 226573, 40051, 226630],
            'filtering': {'valid': F.col('value').between(0, 4000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'EBL',
            'codes': [40064, 226626, 40491, 226629],
            'filtering': {'valid': F.col('value').between(0, 10000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Emesis',
            'codes': [40067, 226571, 40490, 41015, 40427],
            'filtering': {'valid': F.col('value').between(0, 2000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Jackson-Pratt',
            # see get_jackson_pratt_itemids
            'codes': [40072, 40088, 40071, 40092, 41213, 41214, 42824, 42823, 226599, 226600,
                      226602,
                      226601],
            'filtering': {'valid': F.col('value').between(0, 2000),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Residual',
            'codes': [227510, 227511, 42837, 43892, 44909, 44959],
            'filtering': {'valid': F.col('value').between(0, 1050),
                          'action': 'replace_with_median'},
        },
        {
            'name': 'Pre-admission Output',
            'codes': [40060, 226633],
            'filtering': {'valid': F.col('value').between(0, 13000),
                          'action': 'replace_with_median'},
        },
        # INPUTEVENTS
        {
            'name': 'Vasopressin',
            'codes': [30051, 222315],
            # 'select': F.when(  # Fixme: remove this complex selector and just filter outliers out
            #     (F.col('value') == 0)
            #     | ((F.col('unit').isin(['U', 'units']))
            #        & (F.col('value').between(0, 400))),
            #     F.col('value'))
            # .otherwise(2.4), # median
            'filtering': {
                'valid': F.col('value').between(0, 400) & F.col('unit').isin(['U', 'units']),
                'action': 'replace_with_median'
            },
            'unit': 'units',
        },
        {  # TODO: some values have dose unit
            'name': 'Vacomycin',
            'codes': [225798],
            'filtering': {
                'valid': F.col('value').between(0, 8),
                'action': 'replace_with_median'
            },
            'select': F.when(F.col('unit') == 'mg',
                             F.col('value') / 1000).otherwise(F.col('value')),
            'unit': 'g',  # TODO: this has to be converted to g from mg
        },
        {
            'name': 'Calcium Gluconate',
            'codes': [30023, 221456, 227525, 42504, 43070, 45699, 46591, 44346, 46291],
            'select': F.when(F.col('unit') == 'mg',
                             F.col('value') / 1000).otherwise(F.col('value')),
            'unit': 'g',  # TODO: this has to be converted to g from ['mg', 'gm', 'grams']
            'filtering': {
                'valid': F.col('value').between(0, 200) & F.col('unit').isin(['mg', 'gm', 'grams']),
                'action': 'replace_with_median'},
        },
        {
            'name': 'Furosemide',
            'codes': [30123, 221794, 228340],
            'filtering': {'valid': F.col('value').between(0, 250), 'action': 'replace_with_median'},
            # 'range': [
            #     ('value', '>=', 0),
            #     ('value', '<=', 250),
            #     # ('unit', '==', 'mg'), # TODO: or any other if value is 0, but shouldn't we just filter them out?
            # ],
            'units': 'mg',
        },
        {  # boolean type
            'name': 'Famotidine',
            'codes': [225907],
            'filtering': {
                'valid': F.col('value').between(0, 1) \
                         & ((F.col('value') == 0) | (F.col('unit') == 'dose')),
                'action': 'replace_with_median'
            },
            # 'range': [
            #     ('value', '>=', 0),
            #     ('value', '<=', 1),
            #     ('unit', '==', 'dose'),
            # ],
            'unit': 'dose',
        },
        {  # boolean type
            'name': 'Piperacillin',
            'codes': [225893, 225892],
            'filtering': {
                'valid': F.col('value').between(0, 1) \
                         & ((F.col('value') == 0) | (F.col('unit') == 'dose')),
                'action': 'replace_with_median'
            },
        },
        {
            'name': 'Cefazolin',
            'codes': [225850],
            'unit': 'dose',
            'filtering': {
                'valid': F.col('value').between(0, 2) \
                         & ((F.col('value') == 0) | (F.col('unit') == 'dose')),
                'action': 'replace_with_median'
            },
        },
        {
            'name': 'Fiber',
            'codes': [225936, 30166, 30073, 227695, 30088, 225928, 226051, 226050, 226048, 45381,
                      45597,
                      227699, 227696, 44218, 45406, 44675, 226049, 44202, 45370, 227698, 226027,
                      42106,
                      43994, 45865, 44318, 42091, 44699, 44010, 43134, 44045, 43088, 42641, 45691,
                      45515, 45777, 42663, 42027, 44425, 45657, 45775, 44631, 44106, 42116, 44061,
                      44887, 42090, 42831, 45541, 45497, 46789, 44765, 42050],
            'unit': 'ml',
            'filtering': {'valid': F.col('value').between(0, 1600),
                          'action': 'replace_with_median'},
        },
        {  # boolean type
            'name': 'Pantoprazole',
            'codes': [225910, 40549, 41101, 41583, 44008, 40700, 40550],
            'unit': 'dose',
            'select': F.lit(1),
        },
        {
            'name': 'Magnesium Sulphate',
            'codes': [222011, 30027, 227524],
            'unit': 'g',  # TODO: assume unit in ['gm', 'grams', 'mg']
            'filtering': {
                'valid': F.col('value').between(0, 125),
                'action': 'replace_with_median'
            },
        },
        {
            'name': 'KCl',
            'codes': [30026, 225166, 227536],
            'unit': 'mEq',
            'filtering': {'valid': F.col('value').between(0, 501), 'action': 'replace_with_median'},
        },
        {
            'name': 'Heparin',
            'codes': [30025, 225975, 225152],
            'unit': 'units',  # TODO: VALUEUOM.isin(['U', 'units'])
            'filtering': {
                'valid': F.col('value').between(0, 25300) & F.col('unit').isin(['U', 'units']),
                'action': 'replace_with_median',
            }
        },
        {
            'codes': [30124, 221668],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 500)
            },
            'name': 'Midazolam'
        },
        {
            'codes': [30131, 222168],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 12000)
            },
            'name': 'Propofol'
        },
        {
            'codes': [220862, 30009],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 750)
            },
            'name': 'Albumin 25%'
        },
        {
            'codes': [220864, 30008],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1300)
            },
            'name': 'Albumin 5%'
        },
        {
            'codes': [30005, 220970],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 33000)
            },
            'name': 'Fresh Frozen Plasma'
        },
        {
            'codes': [30141, 221385],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 300)
            },
            'name': 'Lorazepam'
        },
        {
            'codes': [30126, 225154],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 4000)
            },
            'name': 'Morphine Sulfate'
        },
        {
            'codes': [30144, 225799],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 7000)
            },
            'name': 'Gastric Meds'
        },
        {
            'codes': [30021, 225828],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 17000)
            },
            'name': 'Lactated Ringers'
        },
        {
            'codes': [30125, 221986],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 50)
            },
            'name': 'Milrinone'
        },
        {
            'codes': [30101, 226364, 30108, 226375],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 22000)
            },
            'name': 'OR/PACU Crystalloid'
        },
        {
            'codes': [30001, 225168, 30104, 226368, 227070],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 17250)
            },
            'name': 'Packed RBC'
        },
        {
            'codes': [30056, 226452, 30109, 226377],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 11000)
            },
            'name': 'PO intake'
        },
        {
            'codes': [30128, 221749, 30127],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1200)
            },
            'name': 'Neosynephrine'
        },
        {
            'codes': [226089, 30063],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1000)
            },
            'name': 'Piggyback'
        },
        {
            'codes': [30121, 222056, 30049],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 350)
            },
            'name': 'Nitroglycerine'
        },
        {
            'codes': [30050, 222051],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 430)
            },
            'name': 'Nitroprusside'
        },
        {
            'codes': [225974],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 151)
            },
            'name': 'Metoprolol'
        },
        {
            'codes': [30120, 221906, 30047],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 80)
            },
            'name': 'Norepinephrine'
        },
        {
            'codes': [30102, 226365, 30107, 226376],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 20000)
            },
            'name': 'Colloid'
        },
        {
            'codes': [221828],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 80)
            },
            'name': 'Hydralazine'
        },
        {
            'codes': [226453, 30059],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 2100)
            },
            'name': 'GT Flush'
        },
        {
            'codes': [30163, 221833],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 125)
            },
            'name': 'Hydromorphone'
        },
        {
            'codes': [225942, 30118, 221744, 30149],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 20)
            },
            'name': 'Fentanyl'
        },
        {
            'codes': [30045, 223258, 30100],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1500)
            },
            'name': 'Insulin Regular'
        },
        {
            'codes': [223262],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 340)
            },
            'name': 'Insulin Humalog'
        },
        {
            'codes': [223260],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 150)
            },
            'name': 'Insulin largine'
        },
        {
            'codes': [223259],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 100)
            },
            'name': 'Insulin NPH'
        },
        {
            'codes': [30140],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1100)
            },
            'name': 'Unknown'
        },
        {
            'codes': [30013, 220949],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 11000)
            },
            'name': 'D5W'
        },
        {
            'codes': [30015,
                      225823,
                      30060,
                      225825,
                      220950,
                      30016,
                      30061,
                      225827,
                      225941,
                      30160,
                      220952,
                      30159,
                      30014,
                      30017,
                      228142,
                      228140,
                      45360,
                      228141,
                      41550],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 4000)
            },
            'name': 'Dextrose Other'
        },
        {
            'codes': [225158, 30018],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 11000)
            },
            'name': 'Normal Saline'
        },
        {
            'codes': [30020, 225159],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 2000)
            },
            'name': 'Half Normal Saline'
        },
        {
            'codes': [225944, 30065],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 10000)
            },
            'name': 'Sterile Water'
        },
        {
            'codes': [30058,
                      225797,
                      41430,
                      40872,
                      41915,
                      43936,
                      41619,
                      42429,
                      44492,
                      46169,
                      42554],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 2500)
            },
            'name': 'Free Water'
        },
        {
            'codes': [225943],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1500)
            },
            'name': 'Solution'
        },
        {
            'codes': [30043, 221662],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1300)
            },
            'name': 'Dopamine'
        },
        {
            'codes': [30119, 221289, 30044],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 100)
            },
            'name': 'Epinephrine'
        },
        {
            'codes': [30112, 221347, 228339, 45402],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1200)
            },
            'name': 'Amiodarone'
        },
        {
            'codes': [30032, 225916, 225917, 30096],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 1600)
            },
            'name': 'TPN'
        },
        {
            'codes': [227523],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 250)
            },
            'name': 'Magnesium Sulfate (Bolus)'
        },
        {
            'codes': [227522],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 500)
            },
            'name': 'KCl (Bolus)'
        },
        {
            'codes': [30054, 226361],
            'filtering': {
                'action': 'replace_with_median',
                'valid': F.col('value').between(0, 30000)
            },
            'name': 'Pre-admission Intake'
        },
    )
