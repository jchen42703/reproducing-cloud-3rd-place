import numpy as np

def combine_segmentation_classification_dfs(df_segmentation, df_classification):
    """
    From: https://www.kaggle.com/bibek777/heng-s-model-inference-kernel
    Removes false positives from a segmentation model sub using classification model predictions.
    """
    df_mask = df_segmentation.fillna("").copy()
    df_label = df_classification.fillna("").copy()
    # do filtering using predictions from classification and segmentation models
    assert(np.all(df_mask["Image_Label"].values == df_label["Image_Label"].values))
    print((df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"] != "").sum() ) #202
    df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"]=""
    return df_mask


def combine_stage1_stage2(df_1, df_2):
    """Combines the stage 1 and stage 2 submissions.

    Replaces the positive class predictions in df_1 with those created in
    df_2.

    Args:
        df_1 (pd.DataFrame): Stage 1 submission
        df_2 (pd.DataFrame): Stage 2 submission
    
    Return:
        combined_df (pd.DataFrame):

    """
    df_1_temp = df_1.fillna("").copy()
    df_2_temp = df_2.fillna("").copy()
    # Checks that all the rows are the same order
    assert(np.all(df_1_temp["Image_Label"].values == \
                  df_2_temp["Image_Label"].values))
    # Sets all predicted pos in stage 1 with those from stage 2
    condition = df_1_temp["EncodedPixels"] != ""
    df_1_temp.loc[condition, "EncodedPixels"] = df_2_temp.loc[condition,
                                                              "EncodedPixels"]
    return df_1_temp
