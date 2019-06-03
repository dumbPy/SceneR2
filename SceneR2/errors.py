class NoMovingRelevantObjectInData(Exception):
    "Raised when the dataset has no scenarios with moving object data as relevant object for training. The passed videos might have pedestrian or stationary object as relevant object in col ABA_typ_SelObj"
    pass
