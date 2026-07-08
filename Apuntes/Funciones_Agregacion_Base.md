# Mean, Count y Sum

1. Cargar el daraframe

        df = pd.read_csv('/datasets/music.csv')

2.  Filtrar todo el df, devuelve todas las columnas y filas que cumplen condicion

        pop_df = df[df
        ['genre'] == 'pop']

3. obtener una columna especifica

        pop_duration = pop_df['total_play']

## Promedio

        mean_duration = pop_duration.mean()
        print(mean_duration)

### Todo en una sola linea:
        mean_duration = df[df['genre'] == 'pop']['total_play'].mean()

## Contar
Hace un recuento del número de filas que cumplen un criterio particular.

        duration_threshold = 180
        filtro = df[df['total_play'] > duration_threshold]
        resultado = filtro['total_play'].count()

### Todo en una sola linea:
        long_songs= df[df['total_play'] > duration_threshold]['total_play'].count()

## Sumar
Este método suma los valores en una columna
especificada.

        filtro = df[df['user_id'] == '42A57CD3']
        resultado = filtro['total_play'].sum()