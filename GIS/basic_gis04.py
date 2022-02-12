import pandas
import folium

map = folium.Map(location=[13.7278956,100.52412349999],zoom_start=13)
# Import csv file
data = pandas.read_csv("https://raw.githubusercontent.com/thassanai/Introduction-to-Python/main/Data/province_th.csv")
city = list(data["City"])  # Province
lat = list(data["LAT"]) #Latitude
long = list(data["LON"]) #Longtitude
#print(city)

fg = folium.FeatureGroup(name="Thailand Map")

for city,lat,long in zip(city,lat,long):
    fg.add_child(folium.Marker(
    location=[lat,long],
    popup=city))

# Add feature to Map
map.add_child(fg)
map.save("GIS/map/ThailandMap.html")
