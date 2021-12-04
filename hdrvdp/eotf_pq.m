function [rgb_linear] = eotf_pq(rgb_pq)
%eotf_pq  SMPTE 2084 EOTF
rgb_pq = cast(rgb_pq,'double');
rgb_pq =rgb_pq/1023;
m1 = 1305/8192;
m2 = 2523/32;
c1 = 107/128;
c2 = 2413/128;
c3 = 2392/128;

pow_term = rgb_pq.^(1/m2);
numer =pow_term -c1;
numer(numer<0)= 0;

rgb_linear = 10000*(numer./(c2-c3*pow_term)).^(1/m1);


end

