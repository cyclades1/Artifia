import pandas as pd


def main():
    df = pd.read_csv("data.csv")
    print(df.to_string()) 
    calories = [420, 380, 390]

    myvar = pd.Series(calories, index=["1","2","3"])



if __name__=="__main__":
    main()