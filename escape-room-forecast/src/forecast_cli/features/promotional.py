import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_promotional_features(
    df: pd.DataFrame,
    promo_col: str | None = "promotion", # Current name after snaking
    num_coupons_col: str | None = "number_of_coupons",
    coupons_text_col: str | None = "coupons",
    gift_voucher_col: str | None = "specific_gift_voucher",
    prepaid_pkg_col: str | None = "prepaid_package",
    known_covariates_list: list | None = None
) -> pd.DataFrame:
    """Adds promotion and discount related features to the DataFrame.

    Args:
        df: Input DataFrame.
        promo_col: Name of the general promotion indicator column.
        num_coupons_col: Name of the column indicating number of coupons.
        coupons_text_col: Name of the column with coupon code text.
        gift_voucher_col: Name of the gift voucher column.
        prepaid_pkg_col: Name of the prepaid package column.
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added promotional features:
            - ft_has_promo
            - ft_uses_coupon_code
            - ft_is_gift_redemption
            - ft_is_prepaid_pkg
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping promotional features.")
        return df

    new_features = []

    # ft_has_promo: 1 if any promotion indicator is present
    df['ft_has_promo'] = 0
    promo_found = False
    if promo_col and promo_col in df.columns and df[promo_col].notna().any():
        df.loc[df[promo_col].notna(), 'ft_has_promo'] = 1
        promo_found = True
    if num_coupons_col and num_coupons_col in df.columns:
        df[num_coupons_col] = pd.to_numeric(df[num_coupons_col], errors='coerce').fillna(0)
        df.loc[df[num_coupons_col] > 0, 'ft_has_promo'] = 1
        promo_found = True
    if coupons_text_col and coupons_text_col in df.columns and df[coupons_text_col].notna().any():
        df.loc[df[coupons_text_col].astype(str).str.strip().ne('') & df[coupons_text_col].notna(), 'ft_has_promo'] = 1
        promo_found = True
    if gift_voucher_col and gift_voucher_col in df.columns and df[gift_voucher_col].notna().any():
        df.loc[df[gift_voucher_col].astype(str).str.strip().ne('') & df[gift_voucher_col].notna(), 'ft_has_promo'] = 1
        promo_found = True
    if prepaid_pkg_col and prepaid_pkg_col in df.columns and df[prepaid_pkg_col].notna().any():
        df.loc[df[prepaid_pkg_col].astype(str).str.strip().ne('') & df[prepaid_pkg_col].notna(), 'ft_has_promo'] = 1
        promo_found = True
    
    new_features.append('ft_has_promo')
    if promo_found:
        logger.info(f"Added 'ft_has_promo'. Distribution: {df['ft_has_promo'].value_counts(normalize=True).to_dict()}")
    else:
        logger.info("No source columns for 'ft_has_promo' found or they were all empty. 'ft_has_promo' set to 0.")

    # ft_uses_coupon_code: 1 if 'coupons' text field is non-empty or number_of_coupons > 0
    df['ft_uses_coupon_code'] = 0
    coupon_use_found = False
    if coupons_text_col and coupons_text_col in df.columns and df[coupons_text_col].notna().any():
        df.loc[df[coupons_text_col].astype(str).str.strip().ne('') & df[coupons_text_col].notna(), 'ft_uses_coupon_code'] = 1
        coupon_use_found = True
    if num_coupons_col and num_coupons_col in df.columns: # Already numeric and filled NA with 0
        df.loc[df[num_coupons_col] > 0, 'ft_uses_coupon_code'] = 1
        coupon_use_found = True
    
    new_features.append('ft_uses_coupon_code')
    if coupon_use_found:
        logger.info(f"Added 'ft_uses_coupon_code'. Distribution: {df['ft_uses_coupon_code'].value_counts(normalize=True).to_dict()}")
    else:
        logger.info("No source columns for 'ft_uses_coupon_code' found or they were all empty. 'ft_uses_coupon_code' set to 0.")

    # ft_is_gift_redemption: 1 if 'specific_gift_voucher' is non-empty
    df['ft_is_gift_redemption'] = 0
    if gift_voucher_col and gift_voucher_col in df.columns and df[gift_voucher_col].notna().any():
        df.loc[df[gift_voucher_col].astype(str).str.strip().ne('') & df[gift_voucher_col].notna(), 'ft_is_gift_redemption'] = 1
        logger.info(f"Added 'ft_is_gift_redemption'. Distribution: {df['ft_is_gift_redemption'].value_counts(normalize=True).to_dict()}")
    else:
        logger.info(f"Source column '{gift_voucher_col}' for 'ft_is_gift_redemption' not found or empty. 'ft_is_gift_redemption' set to 0.")
    new_features.append('ft_is_gift_redemption')

    # ft_is_prepaid_pkg: 1 if 'prepaid_package' is non-empty
    df['ft_is_prepaid_pkg'] = 0
    if prepaid_pkg_col and prepaid_pkg_col in df.columns and df[prepaid_pkg_col].notna().any():
        df.loc[df[prepaid_pkg_col].astype(str).str.strip().ne('') & df[prepaid_pkg_col].notna(), 'ft_is_prepaid_pkg'] = 1
        logger.info(f"Added 'ft_is_prepaid_pkg'. Distribution: {df['ft_is_prepaid_pkg'].value_counts(normalize=True).to_dict()}")
    else:
        logger.info(f"Source column '{prepaid_pkg_col}' for 'ft_is_prepaid_pkg' not found or empty. 'ft_is_prepaid_pkg' set to 0.")
    new_features.append('ft_is_prepaid_pkg')

    if known_covariates_list is not None:
        known_covariates_list.extend(new_features)

    return df 