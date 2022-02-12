import folium

m = folium.Map(location=[17.6334887,100.0913382], zoom_start=13)
folium.Marker(
    [17.6356345,100.0921328], popup="<i>Uttaradit Rajabhat University</i>"
).add_to(m)
folium.Marker(
    [17.6355671,100.0939999], popup="<b>Test02</b>"
).add_to(m)

m.save("GIS/map/01.html")