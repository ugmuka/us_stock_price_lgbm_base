[株式予想コンペ](https://comp.probspace.com/competitions/us_stock_price)のlightgbmベースモデル

# require
* docker-compose

# build

```
docker-compose build
```

# jupyter notebook

```
docker-compose up
```
起動後、ブラウザで http://localhost:8888 にアクセスしてください。

# スクリプト実行
bash
```
docker-compose run --rm us_stock_price_lgbm bash
```


```
python main.py
```

# 結果確認
`log/`配下にvalidationの実行結果が保存されます。