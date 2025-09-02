#!/usr/bin/env python3

import sys
sys.path.append('/workspace/app')
from app.graphrag import get_graph_db

try:
    graph_db = get_graph_db()
    if graph_db.is_connected():
        with graph_db.driver.session() as session:
            result = session.run('MATCH (e:Entity) RETURN e.name, e.type ORDER BY e.name LIMIT 20')
            print('Sample entities in graph:')
            for record in result:
                name = record['e.name']
                entity_type = record['e.type']
                print(f'  - {name} (Type: {entity_type})')
    else:
        print('Graph DB not connected')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
