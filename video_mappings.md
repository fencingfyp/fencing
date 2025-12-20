
# Video Mappings

This file maps assigned video identifiers to their corresponding original match names for reference and organization purposes.

| Assigned Video Name | Original Match Name | Source (if applicable)
|---|---|---|
| foil_1 | Unknown | - |
| foil_3 | Unknown | - |
| foil_4 | Unknown | - |
| epee_1 | Unknown | - |
| epee_2 | Unknown | - |
| sabre_1 | 2026 184 T64 09 M S Individual Ghent BEL FINAL MUELLER GER vs KUERBIS GER | https://www.youtube.com/watch?v=sFzbWmzTBT8 |
| epee_3 | 2025 Heidenheim CDE Womens Epee T16 Germany vs Switzerland | https://www.youtube.com/watch?v=19CjHpfPO0s |
| sabre_2 | 2024 1431 T16 06 F S Individual Tunis TUN GP GREEN NOUTCHA FRA vs CHOI KOR| https://www.youtube.com/watch?v=IdjZ97NRqHQ |
| epee_4 | 2022 80 T64 03 F E Individual Tallinn EST WC BLUE MACKINNON CAN vs BROVKO UKR | https://www.youtube.com/watch?v=20ZVCP7ktMw |
| sabre_3 | Dormagen Day02 Piste Blue (1st match) | https://www.youtube.com/watch?v=5tZXUss0Tic |
| sabre_4 | 2023 1431 T64 10 F S Individual Tunis TUN GP 6 FUSETTI ITA vs CHAMBERLAIN USA | https://www.youtube.com/watch?v=5fVNRwCvoys |


## Useful commands for downloading and processing videos from youtube

```
yt-dlp -f "bv*" "video_url"
```

```
ffmpeg -ss 00:12:00 -to 00:18:30 -i input.mp4 -c copy bout.mp4
```