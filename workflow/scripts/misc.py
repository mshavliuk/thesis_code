from functools import reduce

from pyspark.sql import (
    DataFrame,
    functions as F,
)


def get_urine_itemids(outputevents: DataFrame, items: DataFrame) -> list:
    """
    Selection of Urine events from the MIMIC-III dataset as it was done by STraTS.
    :param d_labitems:
    :return:
    """
    d_labitems = (items
                  .join(outputevents.select('ITEMID').distinct(), on='ITEMID', how='inner')
                  .select(F.lower(F.col('LABEL')).alias('LABEL'), F.col('ITEMID')))
    keys = ['urine', 'foley', 'void', 'nephrostomy', 'condom', 'drainage bag']
    condition = reduce(lambda a, b: a | b, [F.col('LABEL').contains(k) for k in keys])
    
    items = d_labitems.filter(condition)
    ids = items.select('ITEMID').distinct().rdd.flatMap(lambda x: x).collect()
    return ids


def get_stool_itemids(outputevents: DataFrame, items: DataFrame) -> list:
    """
    Selection of Stool events from the MIMIC-III dataset as it was done by STraTS.
    :param d_labitems:
    :return:
    """
    d_labitems = (items
                  .join(outputevents.select('ITEMID').distinct(), on='ITEMID', how='inner')
                  .select(F.lower(F.col('LABEL')).alias('LABEL'), F.col('ITEMID')))
    keys = ['stool', 'fecal', 'colostomy', 'ileostomy', 'rectal']
    condition = reduce(lambda a, b: a | b, [F.col('LABEL').contains(k) for k in keys])
    items = d_labitems.filter(condition)
    ids = items.select('ITEMID').distinct().rdd.flatMap(lambda x: x).collect()
    return ids


def get_chest_tube_itemids(outputevents: DataFrame, items: DataFrame) -> list:
    """
    Selection of Chest Tube events from the MIMIC-III dataset as it was done by STraTS.
    :param outputevents:
    :return:
    """
    d_labitems = (items
                  .join(outputevents.select('ITEMID').distinct(), on='ITEMID', how='inner')
                  .select(F.lower(F.col('LABEL')).alias('LABEL'), F.col('ITEMID')))
    items = d_labitems.filter(F.col('LABEL').contains('chest tube'))
    ids = items.select('ITEMID').distinct().rdd.flatMap(lambda x: x).collect()
    return ids + [226593, 226590, 226591, 226595, 226592]


def get_jackson_pratt_itemids(outputevents: DataFrame) -> list:
    """
    Selection of Jackson-Pratt events from the MIMIC-III dataset as it was done by STraTS.
    :param outputevents:
    :return:
    """
    d_labitems = (spark.read.parquet('data/raw/D_ITEMS.parquet')
                  .join(outputevents.select('ITEMID').distinct(), on='ITEMID', how='inner')
                  .select(F.lower(F.col('LABEL')).alias('LABEL'), F.col('ITEMID')))
    items = d_labitems.filter(F.col('LABEL').contains('jackson'))
    ids = items.select('ITEMID').distinct().rdd.flatMap(lambda x: x).collect()
    return ids
