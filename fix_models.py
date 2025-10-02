# fix_models.py
import joblib
import xgboost as xgb

def fix_xgb_model(old_pkl, new_json, new_pkl):
    try:
        print(f"Fixing {old_pkl} ...")
        model = joblib.load(old_pkl)

        if isinstance(model, xgb.XGBClassifier):
            # Save booster to JSON
            model.get_booster().save_model(new_json)
            # Reload clean model
            fixed_model = xgb.XGBClassifier()
            fixed_model.load_model(new_json)
            # Save as new pickle
            joblib.dump(fixed_model, new_pkl)
            print(f"✅ Fixed and saved: {new_pkl}")
        else:
            print(f"⚠ {old_pkl} is not an XGBClassifier")

    except Exception as e:
        print(f"❌ Error fixing {old_pkl}: {e}")


if __name__ == "__main__":
    fix_xgb_model("best_model_tfidf.pkl", "best_model_tfidf.json", "best_model_tfidf_fixed.pkl")
    fix_xgb_model("best_model_word2vec.pkl", "best_model_word2vec.json", "best_model_word2vec_fixed.pkl")
    