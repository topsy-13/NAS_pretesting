# 06/03/2025
**Objetivo:** Encontrar la mejor arquitectura (+ Configuración de hiperarámetros) 
## _¿Cómo intento muchas arquitecturas en muy poco tiempo?_
¿Otra manera de entrenamiento? Evitar backpropagation?

- Entrenar los K mejores modelos hasta el final?

## Tareas
- Diseñar clases que permitan la experimentación y graben los resultados de la misma:

```python
class Benchmark(architecture, strategy, random_seed):
    return (Performance, Learning Curve, Runtime, HPO(batch_size))
    pass
```

**Es muy importante probar la aleatoridad y la consistencia**: Mismo split, diferentes semillas aleatorias de inicialización. ¿Cuál es la mejor? ¿Se puede hacer un ensemble de estas?

- Revisar estructura de directorios y forma consistente de registrarlos ya que los archivos resultantes pueden ser _masivos_ para las combinaciones de semillas y configuraciones o lo que sea.


