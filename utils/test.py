import pstats

# Lade die Profildatei
stats = pstats.Stats('HARNN/profile_results.out')

# Zeige die 10 Funktionen mit der längsten kumulierten Ausführungszeit an
stats.sort_stats('cumulative').print_stats(10)
