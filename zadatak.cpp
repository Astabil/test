//U programu omogućite unos 10 brojeva. Ispišite najmanji i najveći od njih.




#include <stdio.h>

void main ()
{
	int i,broj,min,max;

	printf("\nUnesi 1. broj= ");
	scanf("%d", &broj);

	min=broj;
	max=broj;

	for (i=2; i<=10; i++)
	{
		printf("\nUnesi %d broj= ",i);
		scanf("%d",&broj);

		if (broj<min)
			min=broj;
		if (broj>max)
			max=broj;
	}
	printf("\nNajveci broj je %d", max);
	printf("\nNajmanji broj je %d", min);
	return;
}




