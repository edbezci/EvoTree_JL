using DataFrames, CSV, StatsBase,Printf, TextAnalysis, Random, MLJ,Languages, EvoTrees



STOPWORDS = stopwords(Languages.English());

function ReadData(data)
    df = CSV.read(data, DataFrame)
    println(first(df,5))
    return df
end

function Feature_Extract(data)
    crps = Corpus(StringDocument.(data.text))
    standardize!(crps, StringDocument)
    remove_case!(crps)
    prepare!(crps, strip_punctuation
        | strip_articles
        | strip_pronouns
        | strip_numbers
        | strip_non_letters)
    remove_words!(crps, STOPWORDS)
    stem!(crps)
    update_lexicon!(crps)
    update_inverse_index!(crps)
    m = DocumentTermMatrix(crps)
    X = tf_idf(m)
    X_t = permutedims(X)
    return X_t
    
end

function Classify(data)
    data = ReadData(data)
    #coerce!(data, :label => Multiclass)
    params = EvoTreeRegressor(loss=:L1, α=0.5, metric = :mae,
    nrounds=100, nbins=100,
    λ = 0.5, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
    train, test = partition(data, 0.7, shuffle=true, rng=123,  stratify= data.label)

    ytrain = train.label
    println(size(ytrain))
    ytest  = test.label
    Xtrain = Feature_Extract(train)
    println(size(Xtrain))
    Xtest = Feature_Extract(test)

    model = fit_evotree(params, Xtrain, ytrain, X_eval = Xtest, Y_eval = ytest, print_every_n = 25)
    pred_eval_L1 = EvoTrees.predict(model, Xtest)

    println(pred_eval_L1)
end

data = "data/data.csv"

@time Classify(data)

