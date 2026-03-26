
import pandas as pd, numpy as np

def generate(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        "age_group": np.random.choice(["18-24","25-34","35-44","45+"], n),
        "income_group": np.random.choice(["<25K","25-50K","50K-1L","1L-2L","2L+"], n),
        "city_tier": np.random.choice(["Tier 1","Tier 2","Tier 3"], n),
        "purchase_frequency": np.random.choice(["Never","Rarely","Occasionally","Frequently"], n),
        "preowned_willingness": np.random.choice(["Definitely","Maybe","No"], n),
        "auth_importance": np.random.choice(["Low","Medium","High"], n),
        "discount_threshold": np.random.choice(["0%","10-20%","20-40%","40%+"], n),
        "trust_score": np.round(np.random.uniform(1,5,n),2),
        "max_spend": np.random.randint(5000,300000,n),
        "luxe_loop_interest": np.random.choice([0,1], n)
    })
    df.to_csv("cleaned_dataset.csv", index=False)

if __name__=="__main__":
    generate()
