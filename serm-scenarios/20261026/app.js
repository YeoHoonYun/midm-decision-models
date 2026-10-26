'use strict';
const input=document.querySelector('#company-search');
if(input){input.addEventListener('input',()=>{const q=input.value.trim().toLowerCase();let count=0;document.querySelectorAll('[data-company]').forEach(row=>{const show=row.dataset.company.toLowerCase().includes(q);row.hidden=!show;if(show)count++;});document.querySelector('#result-count').textContent=`${count}개 기업`;document.querySelector('#no-results').hidden=count>0;});}
document.querySelectorAll('[data-print]').forEach(b=>b.addEventListener('click',()=>window.print()));
