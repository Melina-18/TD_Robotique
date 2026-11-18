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

Le LQR (Linear Quadratic Regulator) est un contrô leur par retour d'é tat qui cherche à amener ou maintenir un systè me à l'é quilibre, le plus efficacement possible. La loi de contrô le est un retour d’é tat :
U(t)=-K*x(t)
On a une fonctionne de cout
𝐽= ∫(𝑥𝑇𝑄𝑥+𝑢𝑇𝑅𝑢)𝑑𝑡∞0
La solution optimale est un retour d'é tat 𝑢=𝑢̿−𝐾(𝑥−𝑥̿), avec 𝐾 = 𝑅⁻¹𝐵ᵀ𝑃, où 𝑃 est la solution de l'é quation algé brique de Riccati :
𝐴𝑇𝑃+𝑃𝐴−𝑃𝐵𝑅−1𝐵𝑇𝑃+𝑄=0
Pour utiliser le LQR on a besoin d’un modè le mathé matique du robot. Nous n'avons pas construit les modè les du robot nous-mê mes : ils viennent de la bibliothè que lerobot-kinematics :
-
Modèle cinématique. get_robot('so100'), dans lerobot_Kinematics.py.
-
Modè le MuJoCo. so_100.xml et scene.xml.
Modèle y = Ax + Bu. L'é tat contient les 6 positions et les 6 vitesses articulaires. M est la matrice d'inertie lue dans MuJoCo dans la posture initiale et D l'amortissement des articulations

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







