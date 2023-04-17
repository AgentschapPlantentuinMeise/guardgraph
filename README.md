# guardgraph

## Introduction

Species interaction graph analysis package for the European GUARDEN
project.

To run this project:

    docker compose up

When start-up is completed, visit http://localhost:8000/ 
To stop the application, run `docker compose down`

## Development notes

    docker ps -a # to see container id of python container
    GUARDCON=$(docker ps -a | grep guardgraph-web | cut -f1 -d' ')
    docker exec -it $GUARDCON /bin/sh

The above code block gives you a shell in the running container. There
you can change dir to /code and work on the code for testing, provided
you clone it to a subdir pf ~/repos

Example:

    %load_ext autoreload
    %autoreload 2
    %cd /code
    from guardgraph.graph import InteractionsGraph
    ig = InteractionsGraph()
    ig.load_interaction_data()

Time to load:

| Entries | Time to load | Std.dev |
| ------- | ------------ | ------- |
| 10      | 37.7 ms      | 4.54 ms |
| 100     | 191 ms       | 23.3 ms |
| 1000    | 1.21 s       | 46.3 ms |
| 10000   | 46.6 s       | 627 ms  |
| 100000  | 21min 28s    |         |

