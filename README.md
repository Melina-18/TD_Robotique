# TD Robotique


## Partie 2 : LeRobot


Voici les vidéos du robot:  

https://github.com/user-attachments/assets/6c097975-c87c-44bb-adbf-c9d0f23e321f

https://github.com/user-attachments/assets/eb477b81-6e69-4ba7-bfd6-5f6a1882882e  

https://github.com/user-attachments/assets/9c53638a-34c6-43a2-958b-94a95acc1355  

https://github.com/user-attachments/assets/7171d93b-7e13-4994-9ac3-b50cdbf5f647  

Le retour a sa position initiale :  

https://github.com/user-attachments/assets/becbd2c6-1f79-417c-afc1-f7b2befdfa70 

## I - LQR

## II - DMP

Un DMP est un système dynamique stable qui « ressort » vers un but, auquel on ajoute un terme appris qui donne sa forme au mouvement. Il a été introduit par Ijspeert, Nakanishi et Schaal (2002, 2013) pour apprendre des gestes par démonstration.
Son intérêt : une seule démonstration suffit, le mouvement converge toujours vers le but, et on peut changer le but, la durée ou le point de départ sans réapprendre.

### II.1 Le système de transformation
Chaque dimension du mouvement (position y) suit un ressort-amortisseur, plus un terme de forçage f :
\tau \dot{v} = \alpha_z \left( \beta_z (g - y) - v \right) + f(s)
\tau \dot{y} = v
• g : le but (position finale) ; y0 : la position de départ ;
• τ : la durée du mouvement (le diviser accélère tout) ;
• αz (« alpha ») : l'amortissement, il règle la rapidité de réaction ;
• βz (« beta ») : la raideur du ressort vers le but.
Sans f, ce système rejoint simplement g. Avec β = α/4, il est amorti de façon critique : il arrive au but le plus vite possible sans le dépasser. Le facteur d'amortissement vaut ζ = ½ × √(α/β) : ζ > 1 est suramorti (lent), ζ < 1 sous-amorti (oscillations).

### II.2 Le système canonique (l'horloge)
Le temps est remplacé par une phase s qui décroît de 1 vers 0 :
\tau \dot{s} = -\alpha_s \, s
Toutes les dimensions partagent cette même phase, ce qui les garde synchronisées. αs règle la vitesse à laquelle s s'éteint.







