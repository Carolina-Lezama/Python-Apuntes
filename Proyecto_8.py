import requests, pandas as pd
from bs4 import BeautifulSoup 
import json

URL = 'https://practicum-content.s3.us-west-1.amazonaws.com/data-analyst-eng/moved_chicago_weather_2017.html'
response = requests.get(URL)
soup=BeautifulSoup(response.text, 'lxml')
table = soup.find('table', attrs={"id": "weather_records"})

heading_table = []
for row in table.find_all('th'):
    heading_table.append(row.text)

content = []
for row in table.find_all('tr'):
    if not row.find_all('th'):  # Solo filas sin encabezados
        content.append([element.text for element in row.find_all('td')])

weather_records = pd.DataFrame(content, columns=heading_table)
print(weather_records)

1.Imprime el campo company_name. Encuentra la cantidad de viajes en taxi para cada compañía de taxis para el 15 y 16 de noviembre de 2017, asigna al campo resultante el nombre trips_amount e imprímelo también. Ordena los resultados por el campo trips_amount en orden descendente.
select
    cabs.company_name as cabs,
    Count(trips.trip_id) as trips_amount 
from
    trips inner join cabs  ON cabs.cab_id = trips.cab_id
where
    trips.start_ts::date = '2017-11-15' or trips.start_ts::date = '2017-11-16'
group by
cabs.company_name
order by
Count(trips.trip_id) desc;

2. 
Encuentra la cantidad de viajes para cada empresa de taxis cuyo nombre contenga las palabras "Yellow" o "Blue" del 1 al 7 de noviembre de 2017. Nombra la variable resultante trips_amount. Agrupa los resultados por el campo company_name.
SELECT 
    cabs.company_name,
    COUNT(trips.trip_id) as trips_amount
        FROM cabs
        INNER JOIN trips ON trips.cab_id = cabs.cab_id
        WHERE cabs.company_name LIKE '%Yellow%'
            AND trips.start_ts::date BETWEEN '2017-11-01' AND '2017-11-07'
        GROUP BY company_name
        
        UNION ALL
        
        SELECT 
            cabs.company_name,
            COUNT(trips.trip_id) as trips_amount  
        FROM cabs
        INNER JOIN trips ON trips.cab_id = cabs.cab_id
        WHERE cabs.company_name LIKE '%Blue%'
            AND trips.start_ts::date BETWEEN '2017-11-01' AND '2017-11-07'
        GROUP BY company_name;

SELECT
    CASE 
        WHEN company_name = 'Flash Cab' THEN 'Flash Cab'
        WHEN company_name = 'Taxi Affiliation Services' THEN 'Taxi Affiliation Services'
        ELSE 'Other'
    END AS company,
    COUNT(trips.trip_id) as trips_amount
FROM cabs
INNER JOIN trips ON trips.cab_id = cabs.cab_id
WHERE trips.start_ts::date BETWEEN '2017-11-01' AND '2017-11-07'
GROUP BY company
ORDER BY trips_amount DESC;

SELECT
    weather_records.ts as ts,
    CASE 
        WHEN weather_records.description like '%rain%' THEN 'Bad'
        WHEN weather_records.description like '%storm%' THEN 'Bad'
        ELSE 'Good'
    END AS weather_conditions
FROM weather_records;

select
    trips.start_ts as start_ts,
    CASE 
        WHEN weather_records.description like '%rain%' THEN 'Bad'
        WHEN weather_records.description like '%storm%' THEN 'Bad'
        ELSE 'Good'
    END AS weather_conditions,
    trips.duration_seconds as duration_seconds
from
    trips
    INNER JOIN weather_records ON weather_records.ts = trips.start_ts   
where
    trips.pickup_location_id = 50 
    and EXTRACT(DOW FROM trips.start_ts) = 6
    and trips.dropoff_location_id = 63
order by
    trips.trip_id;


