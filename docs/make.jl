using JuloVelo
using Documenter

DocMeta.setdocmeta!(JuloVelo, :DocTestSetup, :(using JuloVelo); recursive=true)

makedocs(;
    modules=[JuloVelo],
    authors="Kuan-Chiun Tung <tungsega@yahoo.com.tw>",
    sitename="JuloVelo.jl",
    format=Documenter.HTML(;
        canonical="https://kuanchiun.github.io/JuloVelo.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kuanchiun/JuloVelo.jl",
    devbranch="master",
)
