cmake_minimum_required(VERSION 3.10)
project(SearchSystem_subset C)

set(CMAKE_C_STANDARD 11)

add_executable(DataParser DataParser/DataParser.c
        DataParser/DataParser.h
        DataParser/HashTable.c
        DataParser/HashTable.h)

add_executable(IndexBuilder IndexBuilder/IndexBuilder.c
        IndexBuilder/IndexBuilder.h
        IndexBuilder/MergeHeap.c
        IndexBuilder/MergeHeap.h
        IndexBuilder/InvertedIndex.c
        IndexBuilder/InvertedIndex.h
        IndexBuilder/Lexicon.c
        IndexBuilder/Lexicon.h
        IndexBuilder/Compression.c
        IndexBuilder/Compression.h
        IndexBuilder/Utils.c
        IndexBuilder/Utils.h)

target_link_libraries(IndexBuilder)

add_executable(QueryProcessor QueryProcessor/QueryProcessor.c
        QueryProcessor/QueryProcessor.h
        QueryProcessor/LexiconTable.c
        QueryProcessor/LexiconTable.h
        QueryProcessor/InvertedList.c
        QueryProcessor/InvertedList.h
        QueryProcessor/Decompression.c
        QueryProcessor/Decompression.h
        QueryProcessor/QueryHeap.c
        QueryProcessor/QueryHeap.h)

target_link_libraries(QueryProcessor)
