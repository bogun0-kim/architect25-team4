from langchain_community.tools import DuckDuckGoSearchResults


def get_search_tool():
    # name = 'duckduckgo_results_json'
    name = 'search'
    args_description = f'query="the search query"'
    main_description: str = (
        'A wrapper around Duck Duck Go Search. '
        'Useful for when you need to answer questions about current events. '
        'Input should be a search query.')
    description = f'{name}({args_description}) - a search engine.\n{main_description}'
    return DuckDuckGoSearchResults(name=name, description=description)


if __name__ == '__main__':
    _tool = get_search_tool()
    _result = _tool.invoke("Tokyo current temperature")
    print(_result)
