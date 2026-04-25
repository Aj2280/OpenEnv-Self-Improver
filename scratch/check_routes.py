from server.app import math_app
for route in math_app.routes:
    methods = getattr(route, 'methods', 'WS')
    print(f"Path: {route.path}, Methods: {methods}")
