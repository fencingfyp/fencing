
# Video Mappings

This file maps assigned video identifiers to their corresponding original match names for reference and organization purposes.

| Assigned Video Name | Original Match Name | Source (if applicable)
|---|---|---|
| foil_1 | Unknown | - |
| foil_3 | Unknown | - |
| foil_4 | Unknown | - |
| epee_1 | Unknown | - |
| epee_2 | Unknown | - |
| epee_3 | 2025 Heidenheim CDE Womens Epee T16 Germany vs Switzerland | https://www.youtube.com/watch?v=19CjHpfPO0s |
| sabre_1 | 2025 468 T16 05 F S Individual Seoul KOR GP GREEN KATONA HUN vs POPOVA AIN | https://www.youtube.com/watch?v=I5ZY5d4bfcQ |
| sabre_2 | 2024 1431 T16 06 F S Individual Tunis TUN GP GREEN NOUTCHA FRA vs CHOI KOR| https://www.youtube.com/watch?v=IdjZ97NRqHQ |
| epee_4 | 2022 80 T64 03 F E Individual Tallinn EST WC BLUE MACKINNON CAN vs BROVKO UKR | https://www.youtube.com/watch?v=20ZVCP7ktMw |
| sabre_3 | 
FE W S Individual Plovdiv U23 ZC 2016 wf t16 08 yellow CRISCIO ITA vs OBVINTSEVA RUS | https://www.youtube.com/watch?v=bNtRu3HOy34 |
| sabre_4 | 2023 1431 T64 10 F S Individual Tunis TUN GP 6 FUSETTI ITA vs CHAMBERLAIN USA | https://www.youtube.com/watch?v=5fVNRwCvoys |


## Useful commands

Downloading a video from YouTube
```
yt-dlp -f "bv*" "video_url"
```

Cropping
```
ffmpeg -ss 00:12:00 -to 00:18:30 -i input.mp4 -c copy bout.mp4
```

Converting from H.264 to MPEG-4 codec
```
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k output.mp4
```